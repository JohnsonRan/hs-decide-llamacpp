#!/usr/bin/env python3
"""Latency benchmark for the decide service (POST /v1/systemone).

Standard library only, so it runs on the decide host or inside the container:

    python3 bench.py                                   # reads DECIDE_API_KEY from the environment
    docker compose exec -T decide python - < bench.py  # inside the container, no host Python needed
    python3 bench.py --sizes 500,2000,7000 -n 5 -c 2

Each size is an approximate input-token target; the table reports the server's own count.
"""

import argparse
import json
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
        return time.perf_counter() - t0, data["usage"]["input_tokens"], None
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:200]
        return time.perf_counter() - t0, None, f"HTTP {e.code}: {detail}"
    except Exception as e:  # timeouts, refused connections
        return time.perf_counter() - t0, None, f"{type(e).__name__}: {e}"


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


def run_size(args, size):
    body = build_request(size)
    for _ in range(args.warmup):
        _, _, err = call(args.url, args.key, body, args.timeout)
        if err:
            return None, err
    results, lock = [], threading.Lock()

    def worker(count):
        for _ in range(count):
            r = call(args.url, args.key, body, args.timeout)
            with lock:
                results.append(r)

    per = [args.n // args.concurrency + (1 if i < args.n % args.concurrency else 0)
           for i in range(args.concurrency)]
    t0 = time.perf_counter()
    threads = [threading.Thread(target=worker, args=(c,)) for c in per if c]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0
    ok = [(lat, tok) for lat, tok, err in results if err is None]
    errors = [err for _, _, err in results if err]
    if not ok:
        return None, errors[0]
    lats = [lat for lat, _ in ok]
    tokens = ok[0][1]
    return {
        "tokens": tokens,
        "ok": len(ok),
        "fail": len(errors),
        "min": min(lats),
        "p50": statistics.median(lats),
        "p95": pct(lats, 95),
        "max": max(lats),
        "tok_s": tokens / statistics.median(lats),
        "rps": len(ok) / wall,
    }, (errors[0] if errors else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=os.environ.get("DECIDE_URL", "http://127.0.0.1:8000"),
                    help="base URL without /v1 (default: $DECIDE_URL or http://127.0.0.1:8000)")
    ap.add_argument("--key", default=os.environ.get("DECIDE_API_KEY"), help="default: $DECIDE_API_KEY")
    ap.add_argument("--sizes", default="250,1000,3000,7000",
                    help="comma-separated input-token targets (default: 250,1000,3000,7000)")
    ap.add_argument("-n", type=int, default=5, help="measured requests per size (default: 5)")
    ap.add_argument("-c", "--concurrency", type=int, default=1,
                    help="parallel clients; the server answers one at a time, so >1 shows queueing")
    ap.add_argument("--warmup", type=int, default=1, help="unmeasured requests per size (default: 1)")
    ap.add_argument("--timeout", type=float, default=600, help="per-request seconds (default: 600)")
    ap.add_argument("--json", action="store_true", help="print results as JSON")
    args = ap.parse_args()
    if not args.key:
        sys.exit("No API key: set DECIDE_API_KEY or pass --key")
    args.url = args.url.rstrip("/")
    args.concurrency = max(1, min(args.concurrency, args.n))

    if not args.json:
        print(f"url      {args.url}")
        print(f"version  {get(args.url, args.key, '/version')}")
        print(f"n={args.n} concurrency={args.concurrency} warmup={args.warmup}\n")
        print(f"{'target':>7} {'tokens':>7} {'ok':>4} {'fail':>4} {'min':>8} {'p50':>8} "
              f"{'p95':>8} {'max':>8} {'tok/s':>8} {'req/s':>6}")

    rows = []
    for size in (int(s) for s in args.sizes.split(",") if s.strip()):
        stats, err = run_size(args, size)
        rows.append({"target": size, **(stats or {}), "error": err})
        if args.json:
            continue
        if stats is None:
            print(f"{size:>7}  failed: {err}")
            continue
        print(f"{size:>7} {stats['tokens']:>7} {stats['ok']:>4} {stats['fail']:>4} "
              f"{stats['min']:>7.2f}s {stats['p50']:>7.2f}s {stats['p95']:>7.2f}s {stats['max']:>7.2f}s "
              f"{stats['tok_s']:>8.0f} {stats['rps']:>6.2f}")
        if err:
            print(f"         first error: {err}")
    if args.json:
        print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
