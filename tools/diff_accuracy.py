#!/usr/bin/env python3
"""Accuracy and multi-sequence batch consistency harness for POST /v1/systemone.

Standard library only so it runs directly on the decide host or inside containers:

    python3 diff_accuracy.py --record /tmp/ref.json --url http://127.0.0.1:8000
    python3 diff_accuracy.py --compare /tmp/ref.json --url http://127.0.0.1:8001
    python3 diff_accuracy.py --mixed-batch --url http://127.0.0.1:8001
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


def build_suite(max_tier_tokens=None):
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
            "target_tokens": tok,
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
            "target_tokens": tok,
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
            "target_tokens": tok,
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
        "target_tokens": 1500,
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
        "target_tokens": 300,
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

    # --- Regression boundary & long-context multi-question cases ---

    # 15. Near-tie choice (two symmetrically plausible candidates + close 2-option comparison)
    docs_tie = {
        "D00": "Database migration note A: The primary Postgres cluster cutover to the Tokyo node is tentatively targeting Tuesday at 03:00 UTC with an expected brief read-only pause, pending final failover rehearsal.",
        "D01": "Database migration note B: The primary Postgres cluster cutover to the Tokyo node is tentatively targeting Tuesday at 03:00 UTC with an expected short read-only window, pending final replication sign-off.",
        "D02": EN_TOPICS[3],
        "D03": EN_TOPICS[4],
    }
    cases.append({
        "name": "borderline_choice_en_near_tie",
        "tier": "short_400",
        "target_tokens": 300,
        "body": {
            "model": "hs-decide-1-flash",
            "state": {"query": "Which note gives the more complete Tokyo Postgres cutover schedule?", "documents": docs_tie},
            "questions": {
                "pick_tie": {
                    "type": "choice",
                    "instructions": "Which document best answers `query`?",
                    "criteria": {k: None for k in docs_tie},
                },
                "pair_tie": {
                    "type": "choice",
                    "instructions": "Between D00 and D01 only, which note is more definitive about the Tokyo cutover?",
                    "criteria": {"D00": docs_tie["D00"], "D01": docs_tie["D01"]},
                },
            },
        },
    })

    # 16. Hindsight 5-option page-impact gate (English indirect dependency)
    page_impact_opts = {
        "yes": "Yes, it concerns the page's topic or something it depends on",
        "maybe": "Possibly, it might affect the topic or something it relies on",
        "unsure": "Unclear whether it bears on the topic",
        "depends": "It concerns something the topic relies on (a vendor, dependency, date, person or system)",
        "no": "No, a different subject altogether: nothing this page covers or relies on",
    }
    page_impact_instr = (
        "Is the new memory about the page's topic, or about anything the topic depends on (a "
        "dependency, vendor, date, person or system the topic relies on)? A memory that could "
        "threaten, block, delay or change what the topic is about counts, even when it never names "
        "the topic. Answer no only when it concerns a different subject altogether."
    )
    filler_en_page = "\n".join(EN_TOPICS[:8] * 4)
    cases.append({
        "name": "borderline_choice_en_page_impact",
        "tier": "med_1300",
        "target_tokens": 1100,
        "body": {
            "model": "hs-decide-1-flash",
            "state": (
                "Page topic: Checkout v2 launch risks\n\n"
                "New memory: The third-party payment SDK vendor announced end-of-support on September 1.\n\n"
                "The page as it reads now:\n"
                "Target launch date is September 15. Checkout relies on our billing microservice and hosted card tokenization.\n\n"
                + filler_en_page
            ),
            "questions": {
                "impact": {
                    "type": "choice",
                    "instructions": page_impact_instr,
                    "criteria": page_impact_opts,
                },
            },
        },
    })

    # 17. Hindsight 5-option page-impact gate (Chinese indirect dependency)
    filler_zh_page = "\n".join(ZH_TOPICS[:8] * 5)
    cases.append({
        "name": "borderline_choice_zh_page_impact",
        "tier": "med_1300",
        "target_tokens": 1100,
        "body": {
            "model": "hs-decide-1-flash",
            "state": (
                "Page topic: 9 月 15 日结账系统上线风险\n\n"
                "New memory: 上游支付网关 SDK 供应商宣布将于 9 月 1 日停止旧版签名接口维护，但新接口兼容层尚未排期。\n\n"
                "The page as it reads now:\n"
                "当前上线窗口定在 9 月 15 日，结账系统强依赖外部支付网关与风控回调。\n\n"
                + filler_zh_page
            ),
            "questions": {
                "impact": {
                    "type": "choice",
                    "instructions": page_impact_instr,
                    "criteria": page_impact_opts,
                },
            },
        },
    })

    # 18. Borderline noul near 0.5 (English hedged/partial evidence)
    cases.append({
        "name": "borderline_noul_en_ambiguous",
        "tier": "short_400",
        "target_tokens": 300,
        "body": {
            "model": "hs-decide-1-flash",
            "state": {
                "note": (
                    "During the architecture sync, engineering noted that adding a read replica might ease morning p95 latency, "
                    "and tentatively penciled in an evaluation for Q3 depending on budget sign-off. No hardware order or cutover "
                    "date has been finalized, though team leads generally favored testing a smaller instance first."
                )
            },
            "questions": {
                "budget_approved": {
                    "type": "noul",
                    "instructions": "Does the note confirm that funding for the read replica is effectively secured?",
                },
                "migration_scheduled": {
                    "type": "noul",
                    "instructions": "Is the team likely going ahead with the read replica work next quarter?",
                },
                "latency_mentioned": {
                    "type": "noul",
                    "instructions": "Does the note mention morning p95 latency as a motivation?",
                },
                "oracle_selected": {
                    "type": "noul",
                    "instructions": "Did engineering select Oracle Autonomous Database as the target engine?",
                },
            },
        },
    })

    # 19. Borderline noul near 0.5 (Chinese hedged plan)
    cases.append({
        "name": "borderline_noul_zh_ambiguous",
        "tier": "short_400",
        "target_tokens": 300,
        "body": {
            "model": "hs-decide-1-flash",
            "state": {
                "note": (
                    "团队倾向于在下季度启动缓存层迁移评估，初步看好像能降低早高峰数据库读压，但尚未正式立项，"
                    "最终是否切换还要等压测数据与成本核算结果。"
                )
            },
            "questions": {
                "confirmed_migration": {
                    "type": "noul",
                    "instructions": "根据这段记录，缓存层迁移大概率会在下季度正式落地实施吗？",
                },
                "will_complete_soon": {
                    "type": "noul",
                    "instructions": "团队是否已经具备足够依据直接跳过压测并开始切换？",
                },
                "evaluating_next_quarter": {
                    "type": "noul",
                    "instructions": "记录是否提到团队倾向于在下季度评估缓存层迁移？",
                },
                "already_abandoned": {
                    "type": "noul",
                    "instructions": "团队是否已经明确决定永久放弃所有缓存改造？",
                },
            },
        },
    })

    # 20. Score boundary: Hindsight sufficiency (argmax) + listwise cut (round(score))
    sufficiency_levels = [
        "The evidence does not answer the question",
        "The evidence answers part of the question; something it asks is missing",
        "The evidence fully answers the question",
    ]
    cut_levels = [
        "Only the first candidate is relevant",
        "The first two are relevant",
        "The first three are relevant",
        "The first five are relevant",
        "The first ten are relevant",
        "All of the listed candidates are relevant",
    ]
    cases.append({
        "name": "borderline_score_en_sufficiency",
        "tier": "med_1300",
        "target_tokens": 1100,
        "body": {
            "model": "hs-decide-1-flash",
            "state": (
                "Question: When does the TLS certificate rotation happen, and which service owner triggers it?\n\n"
                "Evidence:\n"
                "[1] Internal gateway certs rotate automatically 30 days before expiry using ACME DNS-01.\n"
                "[2] Gateway-03 experienced an incident on Nov 12 after an internal CA cert expired.\n"
                "[3] The platform security team owns the shared PKI automation scripts, though manual overrides require SRE on-call approval.\n"
                "[4] Redis cache TTL was raised from 60 seconds to 300 seconds.\n"
                "[5] Backups run nightly via pg_dump.\n"
                "[6] Deploy pipeline moved to GitHub Actions.\n\n"
                + filler_en_page
            ),
            "questions": {
                "sufficient": {
                    "type": "score",
                    "instructions": "Does the evidence above answer the question? Judge only what the evidence states, not what could be guessed from it.",
                    "criteria": sufficiency_levels,
                },
                "depth": {
                    "type": "score",
                    "instructions": "How far down this ranked list does genuine relevance to the question extend? Count a candidate as relevant only if it helps answer the question.",
                    "criteria": cut_levels,
                },
            },
        },
    })

    # 21. Score boundary: Chinese multi-level score calibration
    cases.append({
        "name": "borderline_score_zh_levels",
        "tier": "med_1300",
        "target_tokens": 1100,
        "body": {
            "model": "hs-decide-1-flash",
            "state": (
                "问题：Tokyo 主库切换演练的回滚条件是什么，由谁下达指令？\n\n"
                "资料：\n"
                "A. 下周二凌晨 3 点将主库切到东京，只读窗口预计 90 秒。\n"
                "B. 若复制延迟超过 10 秒或错误率连续 2 分钟高于 1%，值班 DBA 可中止切换并切回新加坡。\n"
                + filler_zh_page
            ),
            "questions": {
                "completeness": {
                    "type": "score",
                    "instructions": "资料在多大程度上回答了回滚条件和决策人？",
                    "criteria": ["完全无关", "主题相关但没答案", "部分回答", "完整准确回答"],
                },
                "readiness": {
                    "type": "score",
                    "instructions": "仅凭上述资料，团队对回滚预案的清晰度处于哪个档位？",
                    "criteria": ["毫无头绪", "仅有背景", "部分清晰", "较清晰", "完全清晰可执行"],
                },
            },
        },
    })

    # 22. Long ~6.2k tokens with 24 mixed questions (8 noul, 8 choice, 8 score)
    docs_24q = make_filler_docs(
        EN_TOPICS, 6200, 12, 5,
        "CRITICAL: Staging canary runs on port 9443 with mutual TLS and 15-minute rollback window.",
        "en",
    )
    multi_24_questions = {}
    for qi in range(8):
        multi_24_questions[f"noul_{qi}"] = {
            "type": "noul",
            "instructions": (
                f"Check {qi}: Does `documents.D05` describe staging canary port 9443?"
                if qi % 2 == 0
                else f"Check {qi}: Do the documents specify Kafka 4.0 broker partition counts?"
            ),
        }
        multi_24_questions[f"choice_{qi}"] = {
            "type": "choice",
            "instructions": f"Choice {qi}: Which document mentions staging canary port 9443?",
            "criteria": {f"D{k:02d}": None for k in range(4 + (qi % 5))},
        }
        multi_24_questions[f"score_{qi}"] = {
            "type": "score",
            "instructions": f"Score {qi}: How relevant is `documents.D{(qi + 3) % 12:02d}` to staging canary TLS?",
            "criteria": ["unrelated", "tangential", "partial", "direct"],
        }
    cases.append({
        "name": "max_7000_multi_q_24",
        "tier": "max_7000",
        "target_tokens": 6200,
        "body": {
            "model": "hs-decide-1-flash",
            "state": {"query": "What port and TLS mode does the staging canary use?", "documents": docs_24q},
            "questions": multi_24_questions,
        },
    })

    # 23. Extra-long ~10.5k EN tokens + 12 mixed questions (beyond old 8192 limit)
    docs_11k = make_filler_docs(
        EN_TOPICS, 10500, 16, 10,
        "AUDIT FINDING: Vault token TTL for batch workers was reduced from 24h to 4h in us-east-2.",
        "en",
    )
    xl11_questions = {}
    for qi in range(4):
        xl11_questions[f"noul_{qi}"] = {
            "type": "noul",
            "instructions": (
                f"Fact {qi}: Was Vault token TTL reduced to 4h in us-east-2?"
                if qi < 2
                else f"Fact {qi}: Did the audit ban all batch workers from us-east-2?"
            ),
        }
        xl11_questions[f"choice_{qi}"] = {
            "type": "choice",
            "instructions": f"Locate {qi}: Which document contains the Vault token TTL audit finding?",
            "criteria": {f"D{k:02d}": None for k in range(8, 14)},
        }
        xl11_questions[f"score_{qi}"] = {
            "type": "score",
            "instructions": f"Relevance {qi}: Does `documents.D{10 if qi == 0 else qi:02d}` answer the Vault TTL query?",
            "criteria": ["none", "weak", "partial", "exact"],
        }
    cases.append({
        "name": "xl_11000_en_multi_q",
        "tier": "xl_11k",
        "target_tokens": 10500,
        "body": {
            "model": "hs-decide-1-flash",
            "state": {"query": "What changed for Vault token TTL in us-east-2?", "documents": docs_11k},
            "questions": xl11_questions,
        },
    })

    # 24. Extra-long ~13.5k ZH+code/config tokens + 10 mixed questions (near 16384 limit)
    zh_code_topics = ZH_TOPICS + [
        "services:\n  decide:\n    environment:\n      DECIDE_MAX_QUEUE_SIZE: 32\n      DECIDE_QUEUE_TIMEOUT_S: 60\n",
        "if rendered.tokens() > limit: raise ContextOverflow(f'{rendered.tokens()} > {limit}')\n",
    ]
    docs_14k = make_filler_docs(
        zh_code_topics, 13500, 12, 9,
        "变更确认：网关限流阈值从每分钟 1200 次调整为 600 次，超出返回 HTTP 429 与 Retry-After: 2。",
        "zh",
    )
    xl14_questions = {
        "noul_limit": {"type": "noul", "instructions": "文档是否确认限流阈值调整为每分钟 600 次？"},
        "noul_status": {"type": "noul", "instructions": "触发限流时是否明确返回 HTTP 429 与 Retry-After？"},
        "noul_ambig": {"type": "noul", "instructions": "文档是否已经规定超过 16384 token 时自动拆分重发？"},
        "noul_false": {"type": "noul", "instructions": "文档是否要求完全关闭所有 API Key 鉴权校验？"},
        "choice_doc": {
            "type": "choice",
            "instructions": "哪一份文档包含了网关限流阈值变更说明？",
            "criteria": {k: None for k in docs_14k},
        },
        "choice_code": {
            "type": "choice",
            "instructions": "触发限流时的 HTTP 状态码是哪一个？",
            "criteria": {"400": "Bad Request", "401": "Unauthorized", "429": "Too Many Requests", "503": "Service Unavailable"},
        },
        "choice_impact": {
            "type": "choice",
            "instructions": page_impact_instr,
            "criteria": page_impact_opts,
        },
        "score_hit": {
            "type": "score",
            "instructions": "`documents.D09` 对“网关新限流阈值和状态码”的回答程度如何？",
            "criteria": ["无关", "相关但无答案", "部分回答", "完整准确"],
        },
        "score_miss": {
            "type": "score",
            "instructions": "`documents.D01` 对“网关新限流阈值和状态码”的回答程度如何？",
            "criteria": ["无关", "相关但无答案", "部分回答", "完整准确"],
        },
        "score_partial": {
            "type": "score",
            "instructions": "所有文档整体上对限流配置和运维参数覆盖程度如何？",
            "criteria": ["未覆盖", "少量覆盖", "部分覆盖", "全面覆盖"],
        },
    }
    cases.append({
        "name": "xl_14000_zh_multi_q",
        "tier": "xl_14k",
        "target_tokens": 13500,
        "body": {
            "model": "hs-decide-1-flash",
            "state": {"query": "网关限流阈值、状态码以及相关配置是什么？", "documents": docs_14k},
            "questions": xl14_questions,
        },
    })

    if max_tier_tokens is not None:
        cases = [c for c in cases if c.get("target_tokens", 0) <= max_tier_tokens]
    return cases


def _header_get(headers, name):
    if headers is None:
        return None
    target = name.lower()
    for k, v in headers.items():
        if k.lower() == target:
            return v
    return None


def _parse_pos_int(raw):
    if raw is None or isinstance(raw, bool) or isinstance(raw, float):
        return None
    if isinstance(raw, int):
        return raw if raw >= 1 else None
    try:
        val = int(str(raw).strip())
    except (ValueError, AttributeError):
        return None
    return val if val >= 1 else None


def parse_batch_headers(headers):
    """Extract (batch_size, batch_id) positive integers from X-Decide-Batch-* headers."""
    return (
        _parse_pos_int(_header_get(headers, "X-Decide-Batch-Size")),
        _parse_pos_int(_header_get(headers, "X-Decide-Batch-Id")),
    )


def call_systemone_with_meta(url, key, body, timeout):
    req = urllib.request.Request(
        url + "/v1/systemone",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.load(resp)
            bs, bid = parse_batch_headers(resp.headers)
        return time.perf_counter() - t0, data, {"batch_size": bs, "batch_id": bid}
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:400]
        raise RuntimeError(f"HTTP {e.code} from {url}/v1/systemone: {detail}") from e


def call_systemone(url, key, body, timeout):
    lat, data, _ = call_systemone_with_meta(url, key, body, timeout)
    return lat, data


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
    suite = build_suite(max_tier_tokens=args.max_tier_tokens)
    ver = get_json(args.url, args.key, "/version")
    print(f"Recording {len(suite)} cases from {args.url} ({ver})")
    out_cases = []
    for idx, item in enumerate(suite, 1):
        lat, resp = call_systemone(args.url, args.key, item["body"], args.timeout)
        toks = resp.get("usage", {}).get("input_tokens")
        print(f"  [{idx:02d}/{len(suite)}] {item['name']:32s} tokens={toks} lat={lat:.3f}s")
        out_cases.append({
            "name": item["name"],
            "tier": item["tier"],
            "target_tokens": item.get("target_tokens"),
            "body": item["body"],
            "response": resp,
            "latency": lat,
        })
    payload = {"version": ver, "url": args.url, "cases": out_cases}
    with open(args.record, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print(f"Saved {len(out_cases)} reference responses to {args.record}")


def _top_margin(probs):
    vals = sorted(probs.values(), reverse=True)
    if len(vals) >= 2:
        return vals[0] - vals[1]
    return vals[0] if vals else 1.0


def _is_finite_number(val):
    return isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(val)


def _validate_prob_map(label, ref_probs, cand_probs, *, require_nonempty=True):
    if not isinstance(ref_probs, dict) or (require_nonempty and not ref_probs):
        return f"invalid or empty reference {label} probabilities"
    if not isinstance(cand_probs, dict) or (require_nonempty and not cand_probs):
        return f"missing or invalid candidate {label} probabilities"

    missing_keys = [k for k in ref_probs if k not in cand_probs]
    if missing_keys:
        return f"missing {label} probability keys {missing_keys}"
    extra_keys = sorted(str(k) for k in cand_probs if k not in ref_probs)
    if extra_keys:
        return f"unexpected extra {label} probability keys {extra_keys}"

    bad_ref = [str(k) for k, v in ref_probs.items() if not _is_finite_number(v)]
    if bad_ref:
        return f"invalid reference {label} probability value(s) for {bad_ref}"
    bad_cand = [str(k) for k, v in cand_probs.items() if not _is_finite_number(v)]
    if bad_cand:
        return f"invalid candidate {label} probability value(s) for {bad_cand}"
    return None


def _validate_case_answers(case_name, ref_resp, cand_resp):
    """Strictly verify question IDs, answer types, option sets, and numeric types."""
    schema_errors = []
    valid_pairs = []

    ref_answers = ref_resp.get("answers") if isinstance(ref_resp, dict) else None
    if not isinstance(ref_answers, dict):
        schema_errors.append({
            "case": case_name,
            "qid": "*",
            "reason": "missing or invalid reference answers map",
        })
        return valid_pairs, schema_errors

    cand_answers = cand_resp.get("answers") if isinstance(cand_resp, dict) else None
    if not isinstance(cand_answers, dict):
        schema_errors.append({
            "case": case_name,
            "qid": "*",
            "reason": "missing or invalid candidate answers map",
        })
        return valid_pairs, schema_errors

    for extra_qid in sorted(str(k) for k in cand_answers if k not in ref_answers):
        schema_errors.append({
            "case": case_name,
            "qid": extra_qid,
            "reason": "unexpected extra question in candidate answers",
        })

    for qid, ref_ans in ref_answers.items():
        if not isinstance(ref_ans, dict):
            schema_errors.append({
                "case": case_name,
                "qid": qid,
                "reason": "invalid reference answer object",
            })
            continue
        if qid not in cand_answers or not isinstance(cand_answers[qid], dict):
            schema_errors.append({
                "case": case_name,
                "qid": qid,
                "reason": (
                    "missing question in candidate answers"
                    if qid not in cand_answers
                    else "invalid candidate answer object"
                ),
            })
            continue
        cand_ans = cand_answers[qid]

        qtype = ref_ans.get("type")
        if qtype not in ("choice", "noul", "score"):
            schema_errors.append({
                "case": case_name,
                "qid": qid,
                "reason": f"unknown reference answer type {qtype!r}",
            })
            continue

        cand_type = cand_ans.get("type")
        if cand_type != qtype:
            schema_errors.append({
                "case": case_name,
                "qid": qid,
                "reason": f"answer type mismatch: expected {qtype!r}, got {cand_type!r}",
            })
            continue

        if qtype == "choice":
            ref_probs = ref_ans.get("probabilities")
            cand_probs = cand_ans.get("probabilities")
            prob_err = _validate_prob_map("choice", ref_probs, cand_probs, require_nonempty=True)
            if prob_err is not None:
                schema_errors.append({"case": case_name, "qid": qid, "reason": prob_err})
                continue

            ref_choice = ref_ans.get("choice")
            cand_choice = cand_ans.get("choice")
            if ref_choice is not None and (
                isinstance(ref_choice, bool)
                or not isinstance(ref_choice, (str, int))
                or ref_choice not in ref_probs
            ):
                schema_errors.append({
                    "case": case_name,
                    "qid": qid,
                    "reason": f"invalid reference choice {ref_choice!r}",
                })
                continue
            if ref_choice is not None and cand_choice is None:
                schema_errors.append({
                    "case": case_name,
                    "qid": qid,
                    "reason": "missing candidate choice",
                })
                continue
            if cand_choice is not None and (
                isinstance(cand_choice, bool)
                or not isinstance(cand_choice, (str, int))
                or cand_choice not in cand_probs
            ):
                schema_errors.append({
                    "case": case_name,
                    "qid": qid,
                    "reason": f"invalid candidate choice {cand_choice!r}",
                })
                continue
        elif qtype == "noul":
            ref_noul = ref_ans.get("noul")
            cand_noul = cand_ans.get("noul")
            if not _is_finite_number(ref_noul):
                schema_errors.append({
                    "case": case_name,
                    "qid": qid,
                    "reason": "missing or invalid reference noul value",
                })
                continue
            if not _is_finite_number(cand_noul):
                schema_errors.append({
                    "case": case_name,
                    "qid": qid,
                    "reason": "missing noul value" if cand_noul is None else "invalid non-numeric noul value",
                })
                continue
        elif qtype == "score":
            ref_probs = ref_ans.get("probabilities")
            cand_probs = cand_ans.get("probabilities")
            if ref_probs is not None:
                prob_err = _validate_prob_map("score", ref_probs, cand_probs, require_nonempty=False)
                if prob_err is not None:
                    schema_errors.append({"case": case_name, "qid": qid, "reason": prob_err})
                    continue
            elif cand_probs not in (None, {}):
                schema_errors.append({
                    "case": case_name,
                    "qid": qid,
                    "reason": "unexpected candidate score probabilities",
                })
                continue

            ref_score = ref_ans.get("score")
            cand_score = cand_ans.get("score")
            if ref_score is not None:
                if not _is_finite_number(ref_score):
                    schema_errors.append({
                        "case": case_name,
                        "qid": qid,
                        "reason": "invalid reference score value",
                    })
                    continue
                if cand_score is None:
                    schema_errors.append({
                        "case": case_name,
                        "qid": qid,
                        "reason": "missing candidate score value",
                    })
                    continue
                if not _is_finite_number(cand_score):
                    schema_errors.append({
                        "case": case_name,
                        "qid": qid,
                        "reason": "invalid non-numeric candidate score value",
                    })
                    continue
            elif cand_score is not None and not _is_finite_number(cand_score):
                schema_errors.append({
                    "case": case_name,
                    "qid": qid,
                    "reason": "invalid non-numeric candidate score value",
                })
                continue

        valid_pairs.append((qid, qtype, ref_ans, cand_ans))

    return valid_pairs, schema_errors


def evaluate_comparison(
    ref_cases,
    cand_results,
    *,
    margin_threshold=0.02,
    max_hard_flips=0,
    max_borderline_flips=2,
    max_prob_mae=0.002,
    max_prob_p95=0.005,
    max_prob_diff=0.02,
    max_score_mae=0.005,
):
    if not ref_cases:
        raise ValueError("No reference cases to evaluate")
    if len(ref_cases) != len(cand_results):
        raise ValueError(
            f"Candidate result count ({len(cand_results)}) != reference case count ({len(ref_cases)})"
        )

    choice_total = 0
    choice_match = 0
    noul_total = 0
    noul_match = 0
    score_argmax_total = 0
    score_argmax_match = 0
    score_round_total = 0
    score_round_match = 0

    noul_diffs = []
    score_diffs = []
    score_prob_diffs = []
    choice_prob_diffs = []
    tier_stats = {}
    flips = []
    schema_errors = []
    ref_lats = []
    cand_lats = []
    cov_margin = max(0.05, margin_threshold)
    borderline_evals = 0
    borderline_diffs = []

    def track_boundary(margin, diff):
        nonlocal borderline_evals
        if margin < cov_margin:
            borderline_evals += 1
            if diff is not None:
                borderline_diffs.append(diff)

    def record_flip(case_name, tier, qid, kind, ref_val, ref_prob, cand_val, cand_prob, margin, prob_diff):
        flips.append({
            "case": case_name,
            "tier": tier,
            "qid": qid,
            "kind": kind,
            "ref_val": ref_val,
            "ref_prob": ref_prob,
            "cand_val": cand_val,
            "cand_prob": cand_prob,
            "ref_margin": margin,
            "prob_diff": prob_diff,
            "severity": "hard" if margin >= margin_threshold else "borderline",
        })

    for case, (cand_lat, cand_resp) in zip(ref_cases, cand_results, strict=True):
        ref_resp = case["response"]
        ref_lats.append(case["latency"])
        cand_lats.append(cand_lat)
        tier = case["tier"]
        ts = tier_stats.setdefault(tier, {
            "choice_tot": 0, "choice_ok": 0,
            "noul_flips": 0, "score_flips": 0,
            "prob_diffs": [],
            "ref_lats": [], "cand_lats": [],
        })
        ts["ref_lats"].append(case["latency"])
        ts["cand_lats"].append(cand_lat)

        valid_pairs, case_errors = _validate_case_answers(case["name"], ref_resp, cand_resp)
        schema_errors.extend(case_errors)

        for qid, qtype, ref_ans, cand_ans in valid_pairs:
            if qtype == "choice":
                choice_total += 1
                ts["choice_tot"] += 1
                ref_probs = ref_ans["probabilities"]
                cand_probs = cand_ans["probabilities"]
                ref_choice = ref_ans.get("choice")
                if ref_choice is None:
                    ref_choice = max(ref_probs, key=lambda k: ref_probs[k])
                cand_choice = cand_ans.get("choice")
                if cand_choice is None:
                    cand_choice = max(cand_probs, key=lambda k: cand_probs[k])

                q_diffs = []
                for k, rp in ref_probs.items():
                    d = abs(rp - cand_probs[k])
                    choice_prob_diffs.append(d)
                    ts["prob_diffs"].append(d)
                    q_diffs.append(d)
                margin = _top_margin(ref_probs)
                q_max_diff = max(q_diffs)
                track_boundary(margin, q_max_diff)
                if ref_choice == cand_choice:
                    choice_match += 1
                    ts["choice_ok"] += 1
                else:
                    record_flip(
                        case["name"], tier, qid, "choice",
                        str(ref_choice), ref_probs.get(ref_choice, 0.0),
                        str(cand_choice), cand_probs.get(cand_choice, 0.0),
                        margin, q_max_diff,
                    )
            elif qtype == "noul":
                ref_noul = ref_ans["noul"]
                cand_noul = cand_ans["noul"]
                noul_total += 1
                d = abs(ref_noul - cand_noul)
                noul_diffs.append(d)
                ts["prob_diffs"].append(d)
                margin = abs(ref_noul - 0.5)
                track_boundary(margin, d)
                if (ref_noul >= 0.5) == (cand_noul >= 0.5):
                    noul_match += 1
                else:
                    ts["noul_flips"] += 1
                    record_flip(
                        case["name"], tier, qid, "noul",
                        f"noul={ref_noul:.4f}", ref_noul,
                        f"noul={cand_noul:.4f}", cand_noul,
                        margin, d,
                    )
            elif qtype == "score":
                ref_probs = ref_ans.get("probabilities") or {}
                cand_probs = cand_ans.get("probabilities") or {}

                q_diffs = []
                for k, rp in ref_probs.items():
                    d = abs(rp - cand_probs[k])
                    score_prob_diffs.append(d)
                    ts["prob_diffs"].append(d)
                    q_diffs.append(d)
                q_max_diff = max(q_diffs) if q_diffs else 0.0

                if ref_probs:
                    score_argmax_total += 1
                    ref_top = max(ref_probs, key=lambda k: ref_probs[k])
                    cand_top = max(cand_probs, key=lambda k: cand_probs[k])
                    margin = _top_margin(ref_probs)
                    track_boundary(margin, q_max_diff)
                    if ref_top == cand_top:
                        score_argmax_match += 1
                    else:
                        ts["score_flips"] += 1
                        record_flip(
                            case["name"], tier, qid, "score_argmax",
                            f"lvl={ref_top}", ref_probs[ref_top],
                            f"lvl={cand_top}", cand_probs.get(cand_top, 0.0),
                            margin, q_max_diff,
                        )

                ref_score = ref_ans.get("score")
                cand_score = cand_ans.get("score")
                if ref_score is not None and cand_score is not None:
                    score_diffs.append(abs(ref_score - cand_score))
                    score_round_total += 1
                    margin_r = abs((ref_score - math.floor(ref_score)) - 0.5)
                    track_boundary(margin_r, q_max_diff if q_diffs else None)
                    if round(ref_score) == round(cand_score):
                        score_round_match += 1
                    else:
                        ts["score_flips"] += 1
                        record_flip(
                            case["name"], tier, qid, "score_round",
                            f"round={round(ref_score)}({ref_score:.3f})", ref_score,
                            f"round={round(cand_score)}({cand_score:.3f})", cand_score,
                            margin_r, q_max_diff if q_diffs else abs(ref_score - cand_score),
                        )

    all_prob_diffs = noul_diffs + choice_prob_diffs + score_prob_diffs
    all_prob_mae = statistics.mean(all_prob_diffs) if all_prob_diffs else 0.0
    all_prob_p95 = pct(all_prob_diffs, 95) if all_prob_diffs else 0.0
    all_prob_max = max(all_prob_diffs) if all_prob_diffs else 0.0
    score_mae = statistics.mean(score_diffs) if score_diffs else 0.0

    hard_flips = [f for f in flips if f["severity"] == "hard"]
    borderline_flips = [f for f in flips if f["severity"] == "borderline"]

    checks = [
        {
            "name": "schema_errors",
            "actual": len(schema_errors),
            "limit": 0,
            "passed": len(schema_errors) == 0,
        },
        {
            "name": "hard_flips",
            "actual": len(hard_flips),
            "limit": max_hard_flips,
            "passed": len(hard_flips) <= max_hard_flips,
        },
        {
            "name": "all_prob_mae",
            "actual": all_prob_mae,
            "limit": max_prob_mae,
            "passed": all_prob_mae <= max_prob_mae,
        },
        {
            "name": "all_prob_p95",
            "actual": all_prob_p95,
            "limit": max_prob_p95,
            "passed": all_prob_p95 <= max_prob_p95,
        },
        {
            "name": "all_prob_max",
            "actual": all_prob_max,
            "limit": max_prob_diff,
            "passed": all_prob_max <= max_prob_diff,
        },
        {
            "name": "score_mae",
            "actual": score_mae,
            "limit": max_score_mae,
            "passed": score_mae <= max_score_mae,
        },
    ]
    if max_borderline_flips >= 0:
        checks.insert(
            2,
            {
                "name": "borderline_flips",
                "actual": len(borderline_flips),
                "limit": max_borderline_flips,
                "passed": len(borderline_flips) <= max_borderline_flips,
            },
        )

    tier_rows = {}
    for tier, ts in tier_stats.items():
        r_avg = statistics.mean(ts["ref_lats"])
        c_avg = statistics.mean(ts["cand_lats"])
        tier_rows[tier] = {
            "choice_ok": ts["choice_ok"],
            "choice_tot": ts["choice_tot"],
            "noul_flips": ts["noul_flips"],
            "score_flips": ts["score_flips"],
            "prob_mae": statistics.mean(ts["prob_diffs"]) if ts["prob_diffs"] else 0.0,
            "prob_max": max(ts["prob_diffs"]) if ts["prob_diffs"] else 0.0,
            "ref_avg": r_avg,
            "cand_avg": c_avg,
            "speedup": r_avg / c_avg if c_avg > 0 else 0.0,
        }

    return {
        "cases": len(ref_cases),
        "choice_match": choice_match,
        "choice_total": choice_total,
        "noul_match": noul_match,
        "noul_total": noul_total,
        "score_argmax_match": score_argmax_match,
        "score_argmax_total": score_argmax_total,
        "score_round_match": score_round_match,
        "score_round_total": score_round_total,
        "noul_diffs": {
            "n": len(noul_diffs),
            "mae": statistics.mean(noul_diffs) if noul_diffs else 0.0,
            "p95": pct(noul_diffs, 95) if noul_diffs else 0.0,
            "max": max(noul_diffs) if noul_diffs else 0.0,
        },
        "score_diffs": {
            "n": len(score_diffs),
            "mae": score_mae,
            "p95": pct(score_diffs, 95) if score_diffs else 0.0,
            "max": max(score_diffs) if score_diffs else 0.0,
        },
        "all_prob_diffs": {
            "n": len(all_prob_diffs),
            "mae": all_prob_mae,
            "p95": all_prob_p95,
            "max": all_prob_max,
        },
        "borderline_coverage": {
            "margin_lt": cov_margin,
            "count": borderline_evals,
            "max_diff": max(borderline_diffs) if borderline_diffs else 0.0,
        },
        "latency": {
            "ref_sum": sum(ref_lats),
            "cand_sum": sum(cand_lats),
        },
        "tiers": tier_rows,
        "schema_errors": schema_errors,
        "missing_errors": schema_errors,
        "flips": flips,
        "hard_flips": hard_flips,
        "borderline_flips": borderline_flips,
        "margin_threshold": margin_threshold,
        "gate": {
            "passed": all(c["passed"] for c in checks),
            "checks": checks,
        },
    }


def compare_mode(args):
    with open(args.compare, "r", encoding="utf-8") as fh:
        ref_data = json.load(fh)
    ref_cases = ref_data["cases"]
    if not ref_cases:
        sys.exit(f"Reference file {args.compare} contains no cases")
    cand_ver = get_json(args.url, args.key, "/version")
    print(f"Ref  : {ref_data.get('version')}")
    print(f"Cand : {cand_ver}")
    print(f"Cases: {len(ref_cases)}, concurrency={args.concurrency}\n")

    results = [None] * len(ref_cases)
    worker_errors = []
    lock = threading.Lock()
    next_idx = [0]

    def worker():
        while True:
            with lock:
                i = next_idx[0]
                next_idx[0] += 1
            if i >= len(ref_cases):
                return
            try:
                lat, resp = call_systemone(args.url, args.key, ref_cases[i]["body"], args.timeout)
                results[i] = (lat, resp)
            except Exception as exc:
                with lock:
                    worker_errors.append((ref_cases[i]["name"], str(exc)))

    threads = [threading.Thread(target=worker) for _ in range(max(1, args.concurrency))]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0
    if worker_errors:
        detail = "; ".join(f"{name}: {err}" for name, err in worker_errors[:5])
        sys.exit(f"Candidate request error(s): {detail}")

    report = evaluate_comparison(
        ref_cases,
        results,
        margin_threshold=args.margin_threshold,
        max_hard_flips=args.max_hard_flips,
        max_borderline_flips=args.max_borderline_flips,
        max_prob_mae=args.max_prob_mae,
        max_prob_p95=args.max_prob_p95,
        max_prob_diff=args.max_prob_diff,
        max_score_mae=args.max_score_mae,
    )
    report["cand_wall"] = wall

    print("=== Overall Accuracy Diff ===")
    agree_pct = 100.0 * report["choice_match"] / report["choice_total"] if report["choice_total"] else 100.0
    print(f"Choice Top-1 Agreement : {report['choice_match']}/{report['choice_total']} ({agree_pct:.2f}%)")
    if report["noul_total"]:
        npct = 100.0 * report["noul_match"] / report["noul_total"]
        print(f"Noul (>=0.5) Agreement : {report['noul_match']}/{report['noul_total']} ({npct:.2f}%)")
    if report["score_argmax_total"]:
        spct = 100.0 * report["score_argmax_match"] / report["score_argmax_total"]
        print(f"Score Argmax Agreement : {report['score_argmax_match']}/{report['score_argmax_total']} ({spct:.2f}%)")
    if report["score_round_total"]:
        rpct = 100.0 * report["score_round_match"] / report["score_round_total"]
        print(f"Score Round Agreement  : {report['score_round_match']}/{report['score_round_total']} ({rpct:.2f}%)")

    nd = report["noul_diffs"]
    if nd["n"]:
        print(f"Noul Prob Diff         : MAE={nd['mae']:.5f}  p95={nd['p95']:.5f}  max={nd['max']:.5f} (n={nd['n']})")
    sd = report["score_diffs"]
    if sd["n"]:
        print(f"Score Expected Diff    : MAE={sd['mae']:.5f}  p95={sd['p95']:.5f}  max={sd['max']:.5f} (n={sd['n']})")
    ap = report["all_prob_diffs"]
    if ap["n"]:
        print(f"All Option Probs Diff  : MAE={ap['mae']:.5f}  p95={ap['p95']:.5f}  max={ap['max']:.5f} (n={ap['n']})")
    bc = report["borderline_coverage"]
    print(f"Near-Boundary (<{bc['margin_lt']:.2f})  : {bc['count']} decisions evaluated, max_diff={bc['max_diff']:.5f}")
    lat = report["latency"]
    print(f"Total Lat (sum/wall)   : ref_sum={lat['ref_sum']:.2f}s  cand_sum={lat['cand_sum']:.2f}s  cand_wall={wall:.2f}s")

    print("\n=== By Token Tier ===")
    print(
        f"{'tier':12s} {'choice':>8s} {'flips(n/s)':>11s} {'prob_MAE':>10s} "
        f"{'prob_max':>10s} {'ref_avg':>9s} {'cand_avg':>9s} {'speedup':>8s}"
    )
    for tier, ts in report["tiers"].items():
        flip_ns = f"{ts['noul_flips']}/{ts['score_flips']}"
        print(
            f"{tier:12s} {ts['choice_ok']:>3d}/{ts['choice_tot']:<4d} {flip_ns:>11s} "
            f"{ts['prob_mae']:10.5f} {ts['prob_max']:10.5f} {ts['ref_avg']:8.3f}s {ts['cand_avg']:8.3f}s {ts['speedup']:7.2f}x"
        )

    if report["schema_errors"]:
        print("\n=== Schema / Missing Candidate Field Errors ===")
        for m in report["schema_errors"]:
            print(f"  {m['case']}.{m['qid']}: {m['reason']}")
    for heading, cond_str, bucket in (
        ("Hard Decision Flips", f"margin >= {args.margin_threshold:.3f}", report["hard_flips"]),
        ("Borderline Decision Flips", f"margin < {args.margin_threshold:.3f}", report["borderline_flips"]),
    ):
        if not bucket:
            continue
        print(f"\n=== {heading} ({cond_str}) ===")
        for f in bucket:
            print(
                f"  {f['case']}.{f['qid']} [{f['kind']}]: ref={f['ref_val']} ({f['ref_prob']:.4f}) "
                f"-> cand={f['cand_val']} ({f['cand_prob']:.4f}) "
                f"[margin={f['ref_margin']:.4f}, diff={f['prob_diff']:.5f}]"
            )
    if not report["flips"] and not report["schema_errors"]:
        print("\nNo choice, noul(>=0.5), or score decision flips!")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2)

    if args.check:
        status = "PASS" if report["gate"]["passed"] else "FAIL"
        print(f"\n=== Regression Gate: {status} ===")
        for c in report["gate"]["checks"]:
            mark = "OK" if c["passed"] else "FAIL"
            if isinstance(c["actual"], int):
                print(f"  [{mark:4s}] {c['name']:18s}: {c['actual']} <= {c['limit']}")
            else:
                print(f"  [{mark:4s}] {c['name']:18s}: {c['actual']:.5f} <= {c['limit']:.5f}")
        if not report["gate"]["passed"]:
            sys.exit(1)


def _unpack_result(entry):
    if len(entry) == 3:
        lat, resp, meta = entry
        return lat, resp, (meta if isinstance(meta, dict) else {})
    lat, resp = entry
    return lat, resp, {}


def evaluate_mixed_batch(
    selected,
    serial_results,
    batched_results,
    *,
    max_prob_diff=5e-3,
    max_score_diff=5e-3,
    margin_threshold=0.02,
):
    if len(selected) != len(serial_results) or len(selected) != len(batched_results):
        raise ValueError(
            f"Mismatched counts: selected={len(selected)}, serial={len(serial_results)}, batched={len(batched_results)}"
        )

    schema_errors = []
    hard_choice_flips = []
    observed_max_prob_diff = 0.0
    observed_max_score_diff = 0.0
    max_prob_diff_detail = None
    max_score_diff_detail = None
    req_records = []

    for c, s_entry, b_entry in zip(selected, serial_results, batched_results, strict=True):
        s_lat, s_resp, _ = _unpack_result(s_entry)
        b_lat, b_resp, b_meta = _unpack_result(b_entry)

        valid_pairs, case_errors = _validate_case_answers(c["name"], s_resp, b_resp)
        schema_errors.extend(case_errors)

        for qid, qtype, s_ans, b_ans in valid_pairs:
            if qtype == "noul":
                diff = abs(s_ans["noul"] - b_ans["noul"])
                if diff > observed_max_prob_diff:
                    observed_max_prob_diff = diff
                    max_prob_diff_detail = {
                        "case": c["name"],
                        "qid": qid,
                        "key": None,
                        "serial": s_ans["noul"],
                        "batched": b_ans["noul"],
                        "diff": diff,
                    }
            else:
                s_probs = s_ans.get("probabilities") or {}
                b_probs = b_ans.get("probabilities") or {}
                for k, sp in s_probs.items():
                    diff = abs(sp - b_probs[k])
                    if diff > observed_max_prob_diff:
                        observed_max_prob_diff = diff
                        max_prob_diff_detail = {
                            "case": c["name"],
                            "qid": qid,
                            "key": k,
                            "serial": sp,
                            "batched": b_probs[k],
                            "diff": diff,
                        }
                if qtype == "choice":
                    s_choice = s_ans.get("choice")
                    if s_choice is None:
                        s_choice = max(s_probs, key=lambda k: s_probs[k])
                    b_choice = b_ans.get("choice")
                    if b_choice is None:
                        b_choice = max(b_probs, key=lambda k: b_probs[k])
                    margin = _top_margin(s_probs)
                    if s_choice != b_choice and margin >= margin_threshold:
                        hard_choice_flips.append({
                            "case": c["name"],
                            "qid": qid,
                            "serial_choice": str(s_choice),
                            "batched_choice": str(b_choice),
                            "margin": margin,
                        })
                elif qtype == "score":
                    s_score = s_ans.get("score")
                    b_score = b_ans.get("score")
                    if s_score is not None and b_score is not None:
                        s_diff = abs(s_score - b_score)
                        if s_diff > observed_max_score_diff:
                            observed_max_score_diff = s_diff
                            max_score_diff_detail = {
                                "case": c["name"],
                                "qid": qid,
                                "serial": s_score,
                                "batched": b_score,
                                "diff": s_diff,
                            }

        b_tokens = (b_resp.get("usage") or {}).get("input_tokens") if isinstance(b_resp, dict) else None
        s_tokens = (s_resp.get("usage") or {}).get("input_tokens") if isinstance(s_resp, dict) else None
        tokens = _parse_pos_int(b_tokens)
        if tokens is None:
            tokens = _parse_pos_int(s_tokens)

        req_records.append({
            "case": c["name"],
            "tokens": tokens,
            "serial_lat": s_lat,
            "batched_lat": b_lat,
            "batch_size": _parse_pos_int(b_meta.get("batch_size")),
            "batch_id": _parse_pos_int(b_meta.get("batch_id")),
        })

    observed_batches = []
    mixed_batches = []
    batch_id_inferred = False
    coverage_reason = None

    missing_batch_headers = [r["case"] for r in req_records if r["batch_size"] is None]
    if missing_batch_headers:
        covered = False
        coverage_reason = f"missing X-Decide-Batch-Size header on {missing_batch_headers}"
    elif all(r["batch_id"] is not None for r in req_records):
        groups = {}
        for r in req_records:
            groups.setdefault(r["batch_id"], []).append(r)
        for bid, members in groups.items():
            reported_sizes = {m["batch_size"] for m in members}
            consistent = len(reported_sizes) == 1
            if not consistent:
                schema_errors.append({
                    "case": ",".join(m["case"] for m in members),
                    "qid": "*",
                    "reason": f"inconsistent X-Decide-Batch-Size values {sorted(reported_sizes)} for batch_id={bid}",
                })
            reported_bs = next(iter(reported_sizes))
            distinct_tokens = sorted({m["tokens"] for m in members if m["tokens"] is not None})
            is_mixed = (
                consistent
                and reported_bs >= 2
                and len(members) >= 2
                and len(distinct_tokens) >= 2
            )
            info = {
                "batch_id": bid,
                "reported_batch_size": reported_bs,
                "observed_members": len(members),
                "cases": [m["case"] for m in members],
                "distinct_tokens": distinct_tokens,
                "mixed_length": is_mixed,
            }
            observed_batches.append(info)
            if is_mixed:
                mixed_batches.append(info)
        covered = len(mixed_batches) > 0
    elif any(r["batch_id"] is not None for r in req_records):
        missing_bid_cases = [r["case"] for r in req_records if r["batch_id"] is None]
        covered = False
        coverage_reason = f"partial X-Decide-Batch-Id headers (missing on {missing_bid_cases})"
    else:
        batch_id_inferred = True
        batched_members = [r for r in req_records if r["batch_size"] >= 2]
        if batched_members:
            distinct_tokens = sorted({r["tokens"] for r in batched_members if r["tokens"] is not None})
            is_mixed = len(batched_members) >= 2 and len(distinct_tokens) >= 2
            info = {
                "batch_id": None,
                "reported_batch_size": max(r["batch_size"] for r in batched_members),
                "observed_members": len(batched_members),
                "cases": [r["case"] for r in batched_members],
                "distinct_tokens": distinct_tokens,
                "mixed_length": is_mixed,
                "inferred": True,
            }
            observed_batches.append(info)
            if is_mixed:
                mixed_batches.append(info)
        for r in req_records:
            if r["batch_size"] == 1:
                observed_batches.append({
                    "batch_id": None,
                    "reported_batch_size": 1,
                    "observed_members": 1,
                    "cases": [r["case"]],
                    "distinct_tokens": [r["tokens"]] if r["tokens"] is not None else [],
                    "mixed_length": False,
                    "inferred": True,
                })
        covered = len(mixed_batches) > 0

    if not covered and coverage_reason is None:
        if any(b["reported_batch_size"] >= 2 for b in observed_batches):
            coverage_reason = (
                "no mixed-length batch observed (multi-request batches only contained "
                "identical token lengths or a single observed request)"
            )
        else:
            coverage_reason = "no multi-request batch observed (all requests ran serially with batch_size=1)"

    accuracy_ok = (
        not schema_errors
        and not hard_choice_flips
        and observed_max_prob_diff <= max_prob_diff
        and observed_max_score_diff <= max_score_diff
    )
    if not accuracy_ok:
        status = "fail"
    elif not covered:
        status = "uncovered"
    else:
        status = "pass"

    return {
        "status": status,
        "passed": status == "pass",
        "accuracy_ok": accuracy_ok,
        "covered": covered,
        "coverage_reason": coverage_reason,
        "batch_id_inferred": batch_id_inferred,
        "schema_errors": schema_errors,
        "hard_choice_flips": hard_choice_flips,
        "max_prob_diff": observed_max_prob_diff,
        "max_score_diff": observed_max_score_diff,
        "max_prob_diff_detail": max_prob_diff_detail,
        "max_score_diff_detail": max_score_diff_detail,
        "max_prob_diff_limit": max_prob_diff,
        "max_score_diff_limit": max_score_diff,
        "observed_batches": observed_batches,
        "mixed_batches": mixed_batches,
        "requests": req_records,
    }


def mixed_batch_mode(args):
    suite = build_suite()
    # Pick 4 distinct lengths: 400, 1300, 3300, 7000
    selected = []
    for name in ("short_400_en_4opt_start", "med_1300_en_20opt_mid", "long_3300_zh_8opt_end", "max_7000_en_4opt_start"):
        selected.append(next(c for c in suite if c["name"] == name))

    print(f"Testing mixed-length concurrent batch consistency on {args.url} ...")
    serial = []
    for c in selected:
        lat, resp, meta = call_systemone_with_meta(args.url, args.key, c["body"], args.timeout)
        serial.append((lat, resp, meta))
        toks = (resp.get("usage") or {}).get("input_tokens") if isinstance(resp, dict) else None
        print(f"  serial  {c['name']:24s} tokens={toks} lat={lat:.3f}s")

    batched = [None] * len(selected)
    batch_errors = []
    err_lock = threading.Lock()
    barrier = threading.Barrier(len(selected), timeout=10.0)

    def fire(idx):
        try:
            barrier.wait()
            batched[idx] = call_systemone_with_meta(args.url, args.key, selected[idx]["body"], args.timeout)
        except Exception as exc:
            with err_lock:
                batch_errors.append((selected[idx]["name"], str(exc)))

    threads = [threading.Thread(target=fire, args=(i,)) for i in range(len(selected))]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0
    if batch_errors:
        detail = "; ".join(f"{name}: {err}" for name, err in batch_errors)
        sys.exit(f"Mixed-batch request error(s): {detail}")

    report = evaluate_mixed_batch(
        selected,
        serial,
        batched,
        max_prob_diff=args.max_mixed_prob_diff,
        max_score_diff=args.max_mixed_score_diff,
    )
    for r in report["requests"]:
        bs_str = str(r["batch_size"]) if r["batch_size"] is not None else "-"
        bid_str = str(r["batch_id"]) if r["batch_id"] is not None else "-"
        print(
            f"  batched {r['case']:24s} tokens={r['tokens']} "
            f"batch_size={bs_str} batch_id={bid_str} lat={r['batched_lat']:.3f}s"
        )
    print(
        f"Mixed-batch wall={wall:.3f}s (vs sum serial={sum(x[0] for x in serial):.3f}s), "
        f"max_prob_diff={report['max_prob_diff']:.6e}, max_score_diff={report['max_score_diff']:.6e}, "
        f"mixed_batches={len(report['mixed_batches'])}/{len(report['observed_batches'])}"
    )

    if report["status"] == "fail":
        for e in report["schema_errors"]:
            print(f"Schema error {e['case']}.{e['qid']}: {e['reason']}", file=sys.stderr)
        for f in report["hard_choice_flips"]:
            print(
                f"Choice flip {f['case']}.{f['qid']}: {f['serial_choice']} -> {f['batched_choice']} "
                f"(margin={f['margin']:.4f})",
                file=sys.stderr,
            )
        if report["max_prob_diff"] > report["max_prob_diff_limit"] and report["max_prob_diff_detail"]:
            d = report["max_prob_diff_detail"]
            k_part = f".{d['key']}" if d["key"] is not None else ""
            print(
                f"Max prob diff {d['case']}.{d['qid']}{k_part}: "
                f"serial={d['serial']:.6f} vs batched={d['batched']:.6f} "
                f"(diff={d['diff']:.6e} > limit={report['max_prob_diff_limit']:.6e})",
                file=sys.stderr,
            )
        if report["max_score_diff"] > report["max_score_diff_limit"] and report["max_score_diff_detail"]:
            d = report["max_score_diff_detail"]
            print(
                f"Max score diff {d['case']}.{d['qid']}: "
                f"serial={d['serial']:.6f} vs batched={d['batched']:.6f} "
                f"(diff={d['diff']:.6e} > limit={report['max_score_diff_limit']:.6e})",
                file=sys.stderr,
            )
        print("FAIL: mixed-batch schema or accuracy consistency regression!", file=sys.stderr)
        sys.exit(2)
    if report["status"] == "uncovered":
        print(
            f"NOT COVERED: {report['coverage_reason']}. "
            "Ensure server runs with DECIDE_MAX_BATCH_SIZE >= 2 and sufficient DECIDE_BATCH_WAIT_MS.",
            file=sys.stderr,
        )
        sys.exit(3)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=os.environ.get("DECIDE_URL", "http://127.0.0.1:8000"))
    ap.add_argument("--key", default=os.environ.get("DECIDE_API_KEY"))
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("-c", "--concurrency", type=int, default=1)
    ap.add_argument("--max-tier-tokens", type=int, default=None, help="skip suite cases above this target token count when recording")
    ap.add_argument("--check", action="store_true", help="exit 1 if regression gate thresholds fail")
    ap.add_argument("--margin-threshold", type=float, default=0.02, help="min ref margin for a flip to count as hard (default: 0.02)")
    ap.add_argument("--max-hard-flips", type=int, default=0, help="allowed hard decision flips under --check (default: 0)")
    ap.add_argument("--max-borderline-flips", type=int, default=2, help="allowed borderline flips under --check, <0 disables (default: 2)")
    ap.add_argument("--max-prob-mae", type=float, default=0.002, help="max all-option prob MAE under --check (default: 0.002)")
    ap.add_argument("--max-prob-p95", type=float, default=0.005, help="max all-option prob p95 under --check (default: 0.005)")
    ap.add_argument("--max-prob-diff", type=float, default=0.02, help="max single prob diff under --check (default: 0.02)")
    ap.add_argument("--max-score-mae", type=float, default=0.005, help="max expected score MAE under --check (default: 0.005)")
    ap.add_argument("--max-mixed-prob-diff", type=float, default=5e-3, help="max prob diff in --mixed-batch (default: 5e-3)")
    ap.add_argument("--max-mixed-score-diff", type=float, default=5e-3, help="max score diff in --mixed-batch (default: 5e-3)")
    ap.add_argument("--json-out", default=None, help="write comparison report JSON to path")
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
