#!/usr/bin/env python3
"""Accuracy and multi-sequence batch consistency harness for POST /v1/systemone.

Standard library only so it runs directly on the decide host or inside containers:

    python3 diff_accuracy.py --record /tmp/ref.json --url http://127.0.0.1:8000
    python3 diff_accuracy.py --compare /tmp/ref.json --url http://127.0.0.1:8001
    python3 diff_accuracy.py --mixed-batch --url http://127.0.0.1:8001
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

EN_TOPICS = [
    "The deploy pipeline moved from Jenkins to GitHub Actions in March 2026.",
    "Postgres was resized for a host with eight cores and eight gigabytes of memory.",
    "The decision model answers yes-or-no, choice and score questions in one pass.",
    "Backups run nightly to object storage using pg_dump custom format and stay thirty days.",
    "The team agreed to keep the API bound to loopback behind a TLS reverse proxy.",
    "A Tesla P4 build uses CUDA 12.6 because Pascal lacks newer compute kernels.",
    "Consolidation merges duplicate facts and tracks which memories were superseded.",
    "Reflect requests read the entire knowledge page before writing a summary.",
    "Redis cache TTL was raised from 60 seconds to 300 seconds after CPU spiked.",
    "Retain worker slots were reserved so background refreshes cannot starve writes.",
]

ZH_TOPICS = [
    "部署流水线在今年三月从 Jenkins 迁移到了 GitHub Actions，构建耗时减半。",
    "Postgres 数据库从 4 GB 内存节点升级到 8 核 8 GB，开启了 HNSW 索引构建调优。",
    "决策模型可以在单次前向传播里同时回答是非题、单选题和打分题，无需生成文本。",
    "每晚凌晨两点自动执行 pg_dump -Fc 压缩备份并上传到对象存储保留三十天。",
    "服务默认仅监听 127.0.0.1 回环地址，对外暴露必须通过 WireGuard 或 HTTPS 反代。",
    "Tesla P4 属于 Pascal 架构，只支持到 CUDA 12.6，无法使用 vLLM 和 Triton 内核。",
    "记忆整合任务会合并重复事实，并标记哪些旧陈述已经被新信息覆盖失效。",
    "知识页增量刷新默认每条新记忆发一次整页请求，并发为 8，超过上限退回 reflect。",
    "Redis 缓存过期时间从 60 秒上调到 300 秒以缓解早高峰数据库读压力。",
    "为避免刷新队列饿死写入，Retain 单独保留了两个 worker 槽位。",
]


def make_filler_docs(topics, target_tokens, n_docs, needle_idx, needle_text, lang):
    # Empirically ~4.5 chars/token EN, ~1.7 chars/token ZH under Qwen3.5 tokenizer.
    chars_per_tok = 4.5 if lang == "en" else 1.7
    total_chars = int(target_tokens * chars_per_tok)
    per_doc_chars = max(80, total_chars // n_docs)
    docs = {}
    for i in range(n_docs):
        chunks = []
        length = 0
        k = i
        while length < per_doc_chars:
            s = topics[k % len(topics)]
            chunks.append(s)
            length += len(s) + 1
            k += 1
        if i == needle_idx:
            chunks.insert(len(chunks) // 2, needle_text)
        docs[f"D{i:02d}"] = " ".join(chunks)
    return docs


def build_suite():
    cases = []
    tiers = [
        ("short_400", 300),
        ("med_1300", 1100),
        ("long_3300", 3000),
        ("max_7000", 6400),
    ]

    for tier_name, tok in tiers:
        # 1. English 4-option choice + clear yes noul + 4-level score (needle near start)
        docs_en4 = make_filler_docs(
            EN_TOPICS, tok, 4, 0,
            "CRITICAL UPDATE: The production Redis cluster runs on port 6385 with TLS required.",
            "en",
        )
        cases.append({
            "name": f"{tier_name}_en_4opt_start",
            "tier": tier_name,
            "body": {
                "model": "hs-decide-1-flash",
                "state": {"query": "Which port and security setting does production Redis use?", "documents": docs_en4},
                "questions": {
                    "pick": {
                        "type": "choice",
                        "instructions": "Which document specifies the production Redis port?",
                        "criteria": {k: None for k in docs_en4},
                    },
                    "has_tls": {
                        "type": "noul",
                        "instructions": "Does any document state that production Redis requires TLS?",
                    },
                    "no_kafka": {
                        "type": "noul",
                        "instructions": "Do the documents describe upgrading a Kafka cluster to version 4.0?",
                    },
                    "score_hit": {
                        "type": "score",
                        "instructions": "How well does `documents.D00` answer `query`?",
                        "criteria": ["unrelated", "same topic, no answer", "partially helps", "directly answers"],
                    },
                },
            },
        })

        # 2. English 20-option choice (needle in middle) + borderline noul + 5-level score
        docs_en20 = make_filler_docs(
            EN_TOPICS, tok, 20, 11,
            "Incident postmortem: The API outage on Nov 12 was caused by an expired internal CA cert on gateway-03.",
            "en",
        )
        cases.append({
            "name": f"{tier_name}_en_20opt_mid",
            "tier": tier_name,
            "body": {
                "model": "hs-decide-1-flash",
                "state": {"query": "What caused the November 12 API outage?", "documents": docs_en20},
                "questions": {
                    "pick": {
                        "type": "choice",
                        "instructions": "Which document identifies the root cause of the outage?",
                        "criteria": {k: None for k in docs_en20},
                    },
                    "borderline": {
                        "type": "noul",
                        "instructions": "Should the team immediately migrate the entire service to Kubernetes based only on these notes?",
                    },
                    "score_miss": {
                        "type": "score",
                        "instructions": "How well does `documents.D01` answer `query`?",
                        "criteria": ["irrelevant", "tangential", "weak clue", "useful context", "full answer"],
                    },
                },
            },
        })

        # 3. Chinese 8-option choice (needle at end) + noul + score
        docs_zh8 = make_filler_docs(
            ZH_TOPICS, tok, 8, 7,
            "最终决定：下周二凌晨 3 点将主库从新加坡节点切换到东京节点，预计只读窗口 90 秒。",
            "zh",
        )
        cases.append({
            "name": f"{tier_name}_zh_8opt_end",
            "tier": tier_name,
            "body": {
                "model": "hs-decide-1-flash",
                "state": {"query": "主库切到东京的时间和预计只读时长是多少？", "documents": docs_zh8},
                "questions": {
                    "pick": {
                        "type": "choice",
                        "instructions": "哪一份文档给出了主库切换的具体时间和只读窗口？",
                        "criteria": {k: None for k in docs_zh8},
                    },
                    "confirmed": {
                        "type": "noul",
                        "instructions": "文档中是否已经确定主库将迁移到东京节点？",
                    },
                    "contradiction": {
                        "type": "noul",
                        "instructions": "文档是否决定放弃东京节点并迁回法兰克福？",
                    },
                    "score_end": {
                        "type": "score",
                        "instructions": "`documents.D07` 在多大程度上回答了 `query`？",
                        "criteria": ["完全无关", "主题相关但没答案", "部分回答", "完整准确回答"],
                    },
                },
            },
        })

    # Add a few extra compact calibration / tricky borderline & 32-option choice cases
    docs_32 = make_filler_docs(
        EN_TOPICS, 1500, 32, 23,
        "Billing policy update: Enterprise tenants on annual contracts receive a 15% volume discount starting Q2.",
        "en",
    )
    cases.append({
        "name": "extra_32opt_billing",
        "tier": "med_1300",
        "body": {
            "model": "hs-decide-1-flash",
            "state": {"query": "What discount do annual enterprise tenants receive?", "documents": docs_32},
            "questions": {
                "pick": {
                    "type": "choice",
                    "instructions": "Which document gives the exact enterprise discount percentage?",
                    "criteria": {k: None for k in docs_32},
                },
                "ambiguous": {
                    "type": "noul",
                    "instructions": "Do monthly startup tenants also receive the same 15% discount?",
                },
            },
        },
    })

    cases.append({
        "name": "extra_zh_consolidation",
        "tier": "short_400",
        "body": {
            "model": "hs-decide-1-flash",
            "state": {
                "existing_fact": "张明平时习惯喝美式咖啡，基本不喝茶。",
                "new_memory": "张明最近改喝龙井绿茶了，戒掉了咖啡。",
            },
            "questions": {
                "supersedes": {
                    "type": "noul",
                    "instructions": "new_memory 是否更新或推翻了 existing_fact 的偏好？",
                },
                "relation": {
                    "type": "choice",
                    "instructions": "new_memory 与 existing_fact 的关系是？",
                    "criteria": {
                        "duplicate": "表达同一个事实",
                        "contradict": "新事实替代旧偏好",
                        "unrelated": "完全无关",
                    },
                },
                "confidence_score": {
                    "type": "score",
                    "instructions": "两条记录之间的冲突程度有多高？",
                    "criteria": ["无冲突", "轻微差异", "明显冲突", "完全相反"],
                },
            },
        },
    })

    return cases


def call_systemone(url, key, body, timeout):
    req = urllib.request.Request(
        url + "/v1/systemone",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.load(resp)
        return time.perf_counter() - t0, data
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:400]
        raise RuntimeError(f"HTTP {e.code} from {url}/v1/systemone: {detail}") from e


def get_json(url, key, path):
    req = urllib.request.Request(url + path, headers={"Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.load(resp)
    except Exception as e:
        return {"error": str(e)}


def pct(vals, p):
    if not vals:
        return 0.0
    s = sorted(vals)
    return s[min(len(s) - 1, round(p / 100 * (len(s) - 1)))]


def record_mode(args):
    suite = build_suite()
    ver = get_json(args.url, args.key, "/version")
    print(f"Recording {len(suite)} cases from {args.url} ({ver})")
    out_cases = []
    for idx, item in enumerate(suite, 1):
        lat, resp = call_systemone(args.url, args.key, item["body"], args.timeout)
        toks = resp.get("usage", {}).get("input_tokens")
        print(f"  [{idx:02d}/{len(suite)}] {item['name']:24s} tokens={toks} lat={lat:.3f}s")
        out_cases.append({
            "name": item["name"],
            "tier": item["tier"],
            "body": item["body"],
            "response": resp,
            "latency": lat,
        })
    payload = {"version": ver, "url": args.url, "cases": out_cases}
    with open(args.record, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print(f"Saved {len(out_cases)} reference responses to {args.record}")


def compare_mode(args):
    with open(args.compare, "r", encoding="utf-8") as fh:
        ref_data = json.load(fh)
    ref_cases = ref_data["cases"]
    cand_ver = get_json(args.url, args.key, "/version")
    print(f"Ref  : {ref_data.get('version')}")
    print(f"Cand : {cand_ver}")
    print(f"Cases: {len(ref_cases)}, concurrency={args.concurrency}\n")

    results = [None] * len(ref_cases)
    lock = threading.Lock()
    next_idx = [0]

    def worker():
        while True:
            with lock:
                i = next_idx[0]
                next_idx[0] += 1
            if i >= len(ref_cases):
                return
            lat, resp = call_systemone(args.url, args.key, ref_cases[i]["body"], args.timeout)
            results[i] = (lat, resp)

    threads = [threading.Thread(target=worker) for _ in range(max(1, args.concurrency))]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0

    choice_total = 0
    choice_match = 0
    noul_diffs = []
    score_diffs = []
    score_prob_diffs = []
    choice_prob_diffs = []
    tier_stats = {}
    flips = []
    ref_lats = []
    cand_lats = []

    for case, (cand_lat, cand_resp) in zip(ref_cases, results):
        ref_resp = case["response"]
        ref_lats.append(case["latency"])
        cand_lats.append(cand_lat)
        tier = case["tier"]
        ts = tier_stats.setdefault(tier, {
            "choice_tot": 0, "choice_ok": 0, "prob_diffs": [],
            "ref_lats": [], "cand_lats": [],
            "tokens": ref_resp.get("usage", {}).get("input_tokens", 0),
        })
        ts["ref_lats"].append(case["latency"])
        ts["cand_lats"].append(cand_lat)

        for qid, ref_ans in ref_resp["answers"].items():
            cand_ans = cand_resp["answers"][qid]
            qtype = ref_ans["type"]
            if qtype == "choice":
                choice_total += 1
                ts["choice_tot"] += 1
                if ref_ans["choice"] == cand_ans["choice"]:
                    choice_match += 1
                    ts["choice_ok"] += 1
                else:
                    flips.append((
                        case["name"], qid,
                        ref_ans["choice"], ref_ans["probabilities"][ref_ans["choice"]],
                        cand_ans["choice"], cand_ans["probabilities"][cand_ans["choice"]],
                        ref_ans["confidence"],
                    ))
                for k, rp in ref_ans["probabilities"].items():
                    d = abs(rp - cand_ans["probabilities"][k])
                    choice_prob_diffs.append(d)
                    ts["prob_diffs"].append(d)
            elif qtype == "noul":
                d = abs(ref_ans["noul"] - cand_ans["noul"])
                noul_diffs.append(d)
                ts["prob_diffs"].append(d)
                if (ref_ans["noul"] >= 0.5) != (cand_ans["noul"] >= 0.5):
                    flips.append((
                        case["name"], qid,
                        f"noul={ref_ans['noul']:.4f}", ref_ans["noul"],
                        f"noul={cand_ans['noul']:.4f}", cand_ans["noul"],
                        abs(ref_ans["noul"] - 0.5) * 2,
                    ))
            elif qtype == "score":
                score_diffs.append(abs(ref_ans["score"] - cand_ans["score"]))
                for k, rp in ref_ans["probabilities"].items():
                    d = abs(rp - cand_ans["probabilities"][k])
                    score_prob_diffs.append(d)
                    ts["prob_diffs"].append(d)

    all_prob_diffs = noul_diffs + choice_prob_diffs + score_prob_diffs
    print("=== Overall Accuracy Diff ===")
    agree_pct = 100.0 * choice_match / choice_total if choice_total else 100.0
    print(f"Choice Top-1 Agreement : {choice_match}/{choice_total} ({agree_pct:.2f}%)")
    if noul_diffs:
        print(f"Noul Prob Diff         : MAE={statistics.mean(noul_diffs):.5f}  p95={pct(noul_diffs, 95):.5f}  max={max(noul_diffs):.5f} (n={len(noul_diffs)})")
    if score_diffs:
        print(f"Score Expected Diff    : MAE={statistics.mean(score_diffs):.5f}  p95={pct(score_diffs, 95):.5f}  max={max(score_diffs):.5f} (n={len(score_diffs)})")
    if all_prob_diffs:
        print(f"All Option Probs Diff  : MAE={statistics.mean(all_prob_diffs):.5f}  p95={pct(all_prob_diffs, 95):.5f}  max={max(all_prob_diffs):.5f} (n={len(all_prob_diffs)})")
    print(f"Total Lat (sum/wall)   : ref_sum={sum(ref_lats):.2f}s  cand_sum={sum(cand_lats):.2f}s  cand_wall={wall:.2f}s")

    print("\n=== By Token Tier ===")
    print(f"{'tier':12s} {'choice':>8s} {'prob_MAE':>10s} {'prob_max':>10s} {'ref_avg':>9s} {'cand_avg':>9s} {'speedup':>8s}")
    for tier, ts in tier_stats.items():
        r_avg = statistics.mean(ts["ref_lats"])
        c_avg = statistics.mean(ts["cand_lats"])
        spd = r_avg / c_avg if c_avg > 0 else 0.0
        p_mae = statistics.mean(ts["prob_diffs"]) if ts["prob_diffs"] else 0.0
        p_max = max(ts["prob_diffs"]) if ts["prob_diffs"] else 0.0
        print(f"{tier:12s} {ts['choice_ok']:>3d}/{ts['choice_tot']:<4d} {p_mae:10.5f} {p_max:10.5f} {r_avg:8.3f}s {c_avg:8.3f}s {spd:7.2f}x")

    if flips:
        print("\n=== Decision Flips ===")
        for name, qid, r_val, r_p, c_val, c_p, conf in flips:
            print(f"  {name}.{qid}: ref={r_val} ({r_p:.4f}) -> cand={c_val} ({c_p:.4f}) [ref_conf={conf:.4f}]")
    else:
        print("\nNo choice or noul(>=0.5) decision flips!")


def mixed_batch_mode(args):
    suite = build_suite()
    # Pick 4 distinct lengths: 400, 1300, 3300, 7000
    selected = []
    for name in ("short_400_en_4opt_start", "med_1300_en_20opt_mid", "long_3300_zh_8opt_end", "max_7000_en_4opt_start"):
        selected.append(next(c for c in suite if c["name"] == name))

    print(f"Testing mixed-length concurrent batch consistency on {args.url} ...")
    serial = []
    for c in selected:
        lat, resp = call_systemone(args.url, args.key, c["body"], args.timeout)
        serial.append((lat, resp))
        print(f"  serial  {c['name']:24s} tokens={resp['usage']['input_tokens']} lat={lat:.3f}s")

    batched = [None] * len(selected)

    def fire(idx):
        batched[idx] = call_systemone(args.url, args.key, selected[idx]["body"], args.timeout)

    threads = [threading.Thread(target=fire, args=(i,)) for i in range(len(selected))]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0

    max_diff = 0.0
    for c, (_, s_resp), (b_lat, b_resp) in zip(selected, serial, batched):
        for qid, s_ans in s_resp["answers"].items():
            b_ans = b_resp["answers"][qid]
            if s_ans["type"] == "noul":
                max_diff = max(max_diff, abs(s_ans["noul"] - b_ans["noul"]))
            else:
                for k, sp in s_ans["probabilities"].items():
                    max_diff = max(max_diff, abs(sp - b_ans["probabilities"][k]))
        print(f"  batched {c['name']:24s} lat={b_lat:.3f}s")
    print(f"Mixed-batch wall={wall:.3f}s (vs sum serial={sum(x[0] for x in serial):.3f}s), max_prob_diff={max_diff:.6e}")
    if max_diff > 1e-3:
        print("WARNING: non-trivial batch vs serial probability difference!", file=sys.stderr)
        sys.exit(2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=os.environ.get("DECIDE_URL", "http://127.0.0.1:8000"))
    ap.add_argument("--key", default=os.environ.get("DECIDE_API_KEY"))
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("-c", "--concurrency", type=int, default=1)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--record", help="record reference responses to JSON")
    mode.add_argument("--compare", help="compare candidate URL against recorded JSON")
    mode.add_argument("--mixed-batch", action="store_true", help="check mixed-length batch vs serial consistency")
    args = ap.parse_args()
    if not args.key:
        sys.exit("No API key: set DECIDE_API_KEY or pass --key")
    args.url = args.url.rstrip("/")

    if args.record:
        record_mode(args)
    elif args.compare:
        compare_mode(args)
    else:
        mixed_batch_mode(args)


if __name__ == "__main__":
    main()
