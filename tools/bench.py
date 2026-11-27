#!/usr/bin/env python3
"""Latency benchmark for the decide service (POST /v1/systemone).

Standard library only, so it runs on the decide host or inside the container:

    python3 bench.py                                   # reads DECIDE_API_KEY from the environment
    docker compose exec -T decide python - < bench.py  # inside the container, no host Python needed
    python3 bench.py --sizes 500,2000,7000 -n 5 -c 2
    python3 bench.py --mixed --keys "$K0,$K1" --duration 30

Each size is an approximate input-token target; the table reports the server's own count.
"""

import argparse
import json
import math
import os
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request

TOPICS = [
    "The deploy pipeline moved from Jenkins to GitHub Actions in March.",
    "Postgres was resized for a host with four cores and four gigabytes of memory.",
    "The decision model answers yes-or-no, choice and score questions in one pass.",
    "Backups run nightly to object storage and are kept for thirty days.",
    "The team agreed to keep the API bound to loopback behind a reverse proxy.",
    "A Tesla P4 build uses the CUDA 12.6 wheel because Pascal lacks newer kernels.",
    "Consolidation merges duplicate memories and records which facts were superseded.",
    "Reflect requests read the whole knowledge page before writing an answer.",
]


def build_request(target_tokens):
    # About 4 characters per token for this English filler; the server reports the real count.
    docs, chars, i = {}, 0, 0
    while chars < target_tokens * 4:
        text = " ".join(TOPICS[(i + k) % len(TOPICS)] for k in range(4)) + f" (note {i})"
        docs[f"D{i:02d}"] = text
        chars += len(text)
        i += 1
    first = next(iter(docs))
    return {
        "model": "hs-decide-1-flash",
        "state": {"query": "Which change affects the database host?", "documents": docs},
        "questions": {
            "best": {
                "type": "choice",
                "instructions": "Which document best answers `query`?",
                "criteria": {k: None for k in list(docs)[:255]},
            },
            "relevant": {"type": "noul", "instructions": "Does any document mention Postgres?"},
            "score": {
                "type": "score",
                "instructions": f"How well does document `documents.{first}` answer `query`?",
                "criteria": ["about something else", "same topic, no help", "helps", "answers"],
            },
        },
    }


def _header_get(headers, name):
    vals = _header_get_all(headers, name)
    return vals[0] if vals else None


def _header_get_all(headers, name):
    if headers is None:
        return []
    target = name.lower()
    return [v for k, v in headers.items() if k.lower() == target]


def _parse_nonneg_float(raw):
    if raw is None:
        return None
    try:
        val = float(raw.strip())
    except (ValueError, AttributeError):
        return None
    if not math.isfinite(val) or val < 0:
        return None
    return val


def parse_timing_headers(headers):
    """Extract (queue_ms, eval_ms) from X-Decide-* or Server-Timing headers."""
    queue_ms = _parse_nonneg_float(_header_get(headers, "X-Decide-Queue-Ms"))
    eval_ms = _parse_nonneg_float(_header_get(headers, "X-Decide-Eval-Ms"))
    if queue_ms is not None and eval_ms is not None:
        return queue_ms, eval_ms
    for st in _header_get_all(headers, "Server-Timing"):
        for entry in st.split(","):
            parts = [p.strip() for p in entry.split(";") if p.strip()]
            if not parts:
                continue
            metric = parts[0].lower()
            dur = None
            for param in parts[1:]:
                if param.lower().startswith("dur="):
                    dur = _parse_nonneg_float(param[4:])
                    break
            if dur is not None:
                if metric == "queue" and queue_ms is None:
                    queue_ms = dur
                elif metric == "eval" and eval_ms is None:
                    eval_ms = dur
    return queue_ms, eval_ms


def _err_result(t0, status_code, err):
    return {
        "e2e_s": time.perf_counter() - t0,
        "tokens": None,
        "queue_ms": None,
        "eval_ms": None,
        "status_code": status_code,
        "err": err,
    }


def call(url, key, body, timeout):
    req = urllib.request.Request(
        url + "/v1/systemone",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.load(resp)
            queue_ms, eval_ms = parse_timing_headers(resp.headers)
            status_code = getattr(resp, "status", 200)
        return {
            "e2e_s": time.perf_counter() - t0,
            "tokens": data["usage"]["input_tokens"],
            "queue_ms": queue_ms,
            "eval_ms": eval_ms,
            "status_code": status_code,
            "err": None,
        }
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:200]
        return _err_result(t0, e.code, f"HTTP {e.code}: {detail}")
    except Exception as e:  # timeouts, refused connections
        return _err_result(t0, None, f"{type(e).__name__}: {e}")


def get(url, key, path):
    req = urllib.request.Request(url + path, headers={"Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.read().decode(errors="replace").strip()
    except Exception as e:
        return f"unavailable ({e})"


def pct(values, p):
    s = sorted(values)
    return s[min(len(s) - 1, round(p / 100 * (len(s) - 1)))]


def _error_counts(errors):
    counts = {}
    for r in errors:
        if r["status_code"] is not None:
            key = f"HTTP {r['status_code']}"
        else:
            key = (r["err"] or "Error").split(":", 1)[0]
        counts[key] = counts.get(key, 0) + 1
    return counts


def run_size(args, size, key):
    body = build_request(size)
    for _ in range(args.warmup):
        r = call(args.url, key, body, args.timeout)
        if r["err"]:
            return None, r["err"]
    results, lock = [], threading.Lock()

    def worker(count):
        for _ in range(count):
            r = call(args.url, key, body, args.timeout)
            with lock:
                results.append(r)

    per = [
        args.n // args.concurrency + (1 if i < args.n % args.concurrency else 0)
        for i in range(args.concurrency)
    ]
    t0 = time.perf_counter()
    threads = [threading.Thread(target=worker, args=(c,)) for c in per if c]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0

    ok = [r for r in results if r["err"] is None]
    errors = [r for r in results if r["err"] is not None]
    errors_by_code = _error_counts(errors)
    if not ok:
        return None, (errors[0]["err"] if errors else "no requests executed")

    lats = [r["e2e_s"] for r in ok]
    q_vals = [r["queue_ms"] for r in ok if r["queue_ms"] is not None]
    e_vals = [r["eval_ms"] for r in ok if r["eval_ms"] is not None]
    tokens = ok[0]["tokens"]
    return {
        "tokens": tokens,
        "ok": len(ok),
        "fail": len(errors),
        "errors_by_code": errors_by_code,
        "min": min(lats),
        "p50": statistics.median(lats),
        "p95": pct(lats, 95),
        "max": max(lats),
        "q_p50": statistics.median(q_vals) if q_vals else None,
        "q_p95": pct(q_vals, 95) if q_vals else None,
        "e_p50": statistics.median(e_vals) if e_vals else None,
        "e_p95": pct(e_vals, 95) if e_vals else None,
        "tok_s": tokens / statistics.median(lats),
        "rps": len(ok) / wall,
    }, (errors[0]["err"] if errors else None)


def assign_keys(keys, long_c, short_c):
    """Assign API keys to long and short workers, separating tenants when >=2 keys exist."""
    if len(keys) == 1:
        return [(0, keys[0])] * long_c, [(0, keys[0])] * short_c
    long_slots = [(i % len(keys), keys[i % len(keys)]) for i in range(long_c)]
    if long_c < len(keys):
        short_pool = [(i, keys[i]) for i in range(long_c, len(keys))]
    else:
        short_pool = [
            (len(keys) - 1 - (i % len(keys)), keys[len(keys) - 1 - (i % len(keys))])
            for i in range(short_c)
        ]
    short_slots = [short_pool[i % len(short_pool)] for i in range(short_c)]
    return long_slots, short_slots


def summarize_stream(records, window_s):
    ok = [r for r in records if r["err"] is None]
    errors = [r for r in records if r["err"] is not None]
    errors_by_code = _error_counts(errors)
    if not ok:
        return {
            "tokens": 0,
            "total_tokens": 0,
            "ok": 0,
            "fail": len(errors),
            "errors_by_code": errors_by_code,
            "first_error": errors[0]["err"] if errors else None,
            "window_s": window_s,
            "e2e": None,
            "queue_ms": None,
            "eval_ms": None,
            "rps": 0.0,
        }
    lats = [r["e2e_s"] for r in ok]
    q_vals = [r["queue_ms"] for r in ok if r["queue_ms"] is not None]
    e_vals = [r["eval_ms"] for r in ok if r["eval_ms"] is not None]
    tok_vals = [r["tokens"] for r in ok if r["tokens"] is not None]
    return {
        "tokens": int(round(statistics.median(tok_vals))) if tok_vals else 0,
        "total_tokens": sum(tok_vals),
        "ok": len(ok),
        "fail": len(errors),
        "errors_by_code": errors_by_code,
        "first_error": errors[0]["err"] if errors else None,
        "window_s": window_s,
        "e2e": {
            "min": min(lats),
            "p50": statistics.median(lats),
            "p95": pct(lats, 95),
            "max": max(lats),
        },
        "queue_ms": (
            {
                "p50": statistics.median(q_vals),
                "p95": pct(q_vals, 95),
                "max": max(q_vals),
            }
            if q_vals
            else None
        ),
        "eval_ms": (
            {
                "p50": statistics.median(e_vals),
                "p95": pct(e_vals, 95),
            }
            if e_vals
            else None
        ),
        "rps": len(ok) / window_s if window_s > 0 else 0.0,
    }


def _fmt_ms(val):
    return f"{val:.1f}ms" if val is not None else "-"


def run_mixed(args, keys):
    if args.short_concurrency < 1 or args.long_concurrency < 1:
        sys.exit("--short-concurrency and --long-concurrency must be >= 1")
    if args.duration <= 0:
        sys.exit("--duration must be > 0")

    long_slots, short_slots = assign_keys(keys, args.long_concurrency, args.short_concurrency)
    short_body = build_request(args.short_size)
    long_body = build_request(args.long_size)

    for _ in range(args.warmup):
        for body, k in ((short_body, short_slots[0][1]), (long_body, long_slots[0][1])):
            r = call(args.url, k, body, args.timeout)
            if r["err"]:
                if args.json:
                    print(json.dumps({"mode": "mixed", "error": f"Warmup failed: {r['err']}"}, indent=2))
                    sys.exit(1)
                sys.exit(f"Warmup failed: {r['err']}")

    ver = get(args.url, keys[0], "/version")
    short_records, long_records = [], []
    lock = threading.Lock()
    stop_event = threading.Event()
    t0 = 0.0

    def worker(body, key, bucket, interval_s):
        while not stop_event.is_set():
            r = call(args.url, key, body, args.timeout)
            r["finished_s"] = time.perf_counter() - t0
            with lock:
                bucket.append(r)
            if interval_s > 0 and stop_event.wait(interval_s):
                break

    interval_s = max(0.0, args.short_interval_ms / 1000.0)
    threads = [
        threading.Thread(target=worker, args=(long_body, k, long_records, 0.0))
        for _, k in long_slots
    ] + [
        threading.Thread(target=worker, args=(short_body, k, short_records, interval_s))
        for _, k in short_slots
    ]

    t0 = time.perf_counter()
    for t in threads:
        t.start()
    stop_event.wait(args.duration)
    window_s = time.perf_counter() - t0
    stop_event.set()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0

    short_window = max(window_s, max((r["finished_s"] for r in short_records), default=0.0))
    long_window = max(window_s, max((r["finished_s"] for r in long_records), default=0.0))
    short_sum = summarize_stream(short_records, short_window)
    long_sum = summarize_stream(long_records, long_window)
    total_ok = short_sum["ok"] + long_sum["ok"]
    total_fail = short_sum["fail"] + long_sum["fail"]
    total_tokens = short_sum["total_tokens"] + long_sum["total_tokens"]

    report = {
        "mode": "mixed",
        "url": args.url,
        "version": ver,
        "keys_count": len(keys),
        "long_key_slots": [idx for idx, _ in long_slots],
        "short_key_slots": [idx for idx, _ in short_slots],
        "short_target": args.short_size,
        "long_target": args.long_size,
        "short_concurrency": args.short_concurrency,
        "long_concurrency": args.long_concurrency,
        "duration_target_s": args.duration,
        "steady_duration_s": window_s,
        "drain_s": max(0.0, wall - window_s),
        "wall_s": wall,
        "short": short_sum,
        "long": long_sum,
        "aggregate": {
            "total_ok": total_ok,
            "total_fail": total_fail,
            "total_tokens": total_tokens,
            "total_tok_s": total_tokens / wall if wall > 0 else 0.0,
            "total_rps": total_ok / wall if wall > 0 else 0.0,
        },
    }

    if args.json:
        print(json.dumps(report, indent=2))
        if total_ok == 0:
            sys.exit(1)
        return

    print(f"url      {args.url}")
    print(f"version  {ver}")
    print(
        f"mixed    duration={args.duration:.1f}s wall={wall:.2f}s (drain={report['drain_s']:.2f}s) "
        f"keys={len(keys)} (long={args.long_concurrency}@k{','.join(str(i) for i, _ in long_slots)} "
        f"short={args.short_concurrency}@k{','.join(str(i) for i, _ in short_slots)})\n"
    )
    print(
        f"{'stream':<7} {'tokens':>7} {'ok':>4} {'fail':>4} {'e2e_p50':>8} {'e2e_p95':>8} "
        f"{'e2e_max':>8} {'q_p50':>9} {'q_p95':>9} {'eval_p50':>9} {'req/s':>6}"
    )
    for name, s in (("short", short_sum), ("long", long_sum)):
        if s["e2e"] is None:
            print(f"{name:<7} {'-':>7} {s['ok']:>4} {s['fail']:>4}  failed: {s['first_error']}")
            if s["errors_by_code"]:
                print(f"        errors: {s['errors_by_code']}")
            continue
        q50 = _fmt_ms(s["queue_ms"]["p50"]) if s["queue_ms"] else "-"
        q95 = _fmt_ms(s["queue_ms"]["p95"]) if s["queue_ms"] else "-"
        e50 = _fmt_ms(s["eval_ms"]["p50"]) if s["eval_ms"] else "-"
        print(
            f"{name:<7} {s['tokens']:>7} {s['ok']:>4} {s['fail']:>4} "
            f"{s['e2e']['p50']:>7.2f}s {s['e2e']['p95']:>7.2f}s {s['e2e']['max']:>7.2f}s "
            f"{q50:>9} {q95:>9} {e50:>9} {s['rps']:>6.2f}"
        )
        if s["errors_by_code"]:
            print(f"        errors: {s['errors_by_code']}")
    agg = report["aggregate"]
    print(
        f"\ntotal    ok={agg['total_ok']} fail={agg['total_fail']} "
        f"tokens={agg['total_tokens']} throughput={agg['total_tok_s']:.0f} tok/s "
        f"({agg['total_rps']:.2f} req/s)"
    )
    if total_ok == 0:
        sys.exit(1)


def parse_keys(cli_key, env_key, keys_csv):
    keys = []
    seen = set()
    if keys_csv:
        for raw in keys_csv.split(","):
            k = raw.strip()
            if k and k not in seen:
                seen.add(k)
                keys.append(k)
        if cli_key:
            k = cli_key.strip()
            if k and k not in seen:
                keys.insert(0, k)
        return keys
    single = cli_key or env_key
    if single:
        k = single.strip()
        if k:
            keys.append(k)
    return keys


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=os.environ.get("DECIDE_URL", "http://127.0.0.1:8000"),
                    help="base URL without /v1 (default: $DECIDE_URL or http://127.0.0.1:8000)")
    ap.add_argument("--key", default=None, help="default: $DECIDE_API_KEY (ignored when --keys is given)")
    ap.add_argument("--keys", default=None, help="comma-separated API keys for multi-tenant mixed runs")
    ap.add_argument("--sizes", default="250,1000,3000,7000",
                    help="comma-separated input-token targets (default: 250,1000,3000,7000)")
    ap.add_argument("-n", type=int, default=5, help="measured requests per size (default: 5)")
    ap.add_argument("-c", "--concurrency", type=int, default=1,
                    help="parallel clients; the server answers one at a time, so >1 shows queueing")
    ap.add_argument("--warmup", type=int, default=1, help="unmeasured requests per size (default: 1)")
    ap.add_argument("--timeout", type=float, default=600, help="per-request seconds (default: 600)")
    ap.add_argument("--json", action="store_true", help="print results as JSON")
    ap.add_argument("--mixed", action="store_true", help="run concurrent short+long mixed workload")
    ap.add_argument("--short-size", type=int, default=400, help="mixed mode short target tokens (default: 400)")
    ap.add_argument("--long-size", type=int, default=7000, help="mixed mode long target tokens (default: 7000)")
    ap.add_argument("--short-concurrency", type=int, default=1, help="mixed mode short workers (default: 1)")
    ap.add_argument("--long-concurrency", type=int, default=1, help="mixed mode long workers (default: 1)")
    ap.add_argument("--duration", type=float, default=30.0, help="mixed mode measurement seconds (default: 30)")
    ap.add_argument("--short-interval-ms", type=float, default=0.0,
                    help="mixed mode pause between short requests per worker in ms (default: 0)")
    args = ap.parse_args()

    keys = parse_keys(args.key, os.environ.get("DECIDE_API_KEY"), args.keys)
    if not keys:
        sys.exit("No API key: set DECIDE_API_KEY, pass --key, or pass --keys")
    args.url = args.url.rstrip("/")

    if args.mixed:
        run_mixed(args, keys)
        return

    if args.n < 1:
        sys.exit("-n must be >= 1")
    args.concurrency = max(1, min(args.concurrency, args.n))
    primary_key = keys[0]

    if not args.json:
        print(f"url      {args.url}")
        print(f"version  {get(args.url, primary_key, '/version')}")
        print(f"n={args.n} concurrency={args.concurrency} warmup={args.warmup}\n")
        print(
            f"{'target':>7} {'tokens':>7} {'ok':>4} {'fail':>4} {'min':>8} {'p50':>8} "
            f"{'p95':>8} {'max':>8} {'q_p95':>8} {'tok/s':>8} {'req/s':>6}"
        )

    rows = []
    for size in (int(s) for s in args.sizes.split(",") if s.strip()):
        stats, err = run_size(args, size, primary_key)
        rows.append({"target": size, **(stats or {}), "error": err})
        if args.json:
            continue
        if stats is None:
            print(f"{size:>7}  failed: {err}")
            continue
        qp95 = _fmt_ms(stats["q_p95"])
        print(
            f"{size:>7} {stats['tokens']:>7} {stats['ok']:>4} {stats['fail']:>4} "
            f"{stats['min']:>7.2f}s {stats['p50']:>7.2f}s {stats['p95']:>7.2f}s {stats['max']:>7.2f}s "
            f"{qp95:>8} {stats['tok_s']:>8.0f} {stats['rps']:>6.2f}"
        )
        if err:
            print(f"         first error: {err} ({stats['errors_by_code']})")
    if args.json:
        print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
