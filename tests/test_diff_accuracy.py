import json
import socket
import subprocess
import sys
import threading
import time
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import uvicorn

ROOT_DIR = Path(__file__).resolve().parents[1]
TOOLS_DIR = ROOT_DIR / "tools"
SRC_DIR = ROOT_DIR / "src"
for _d in (str(TOOLS_DIR), str(SRC_DIR)):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import diff_accuracy as da  # noqa: E402
import server as srv  # noqa: E402


def test_build_suite_coverage_and_token_filtering():
    suite = da.build_suite()
    assert len(suite) == 24
    names = [c["name"] for c in suite]
    assert len(names) == len(set(names))

    tiers = {c["tier"] for c in suite}
    assert {"short_400", "med_1300", "long_3300", "max_7000", "xl_11k", "xl_14k"} <= tiers

    q24 = next(c for c in suite if c["name"] == "max_7000_multi_q_24")
    assert len(q24["body"]["questions"]) == 24

    for c in suite:
        body = c["body"]
        assert body["model"] == "hs-decide-1-flash"
        assert "state" in body and body["questions"]
        for q in body["questions"].values():
            assert q["type"] in ("noul", "choice", "score")
            if q["type"] == "choice":
                assert isinstance(q["criteria"], dict) and len(q["criteria"]) >= 2
            elif q["type"] == "score":
                assert isinstance(q["criteria"], list) and len(q["criteria"]) >= 2

    filtered = da.build_suite(max_tier_tokens=8000)
    assert len(filtered) == 22
    assert all(c["target_tokens"] <= 8000 for c in filtered)


def _sample_ref_case(name: str, qid: str, ans: dict, tier: str = "short_400") -> dict:
    return {
        "name": name,
        "tier": tier,
        "latency": 0.2,
        "body": {"model": "hs-decide-1-flash", "state": "s", "questions": {qid: {"type": ans["type"], "instructions": "q"}}},
        "response": {"answers": {qid: ans}, "usage": {"input_tokens": 200}},
    }


def _cand_resp(qid: str, ans: dict) -> tuple[float, dict]:
    return (0.1, {"answers": {qid: ans}, "usage": {"input_tokens": 200}})


def test_evaluate_comparison_flip_classification_and_thresholds():
    # 1. Hard choice flip (margin 0.40 >= 0.02) fails gate
    ref = [_sample_ref_case("c1", "q1", {"type": "choice", "choice": "A", "probabilities": {"A": 0.70, "B": 0.30}})]
    cand = [_cand_resp("q1", {"type": "choice", "choice": "B", "probabilities": {"A": 0.49, "B": 0.51}})]
    rep = da.evaluate_comparison(ref, cand)
    assert len(rep["hard_flips"]) == 1
    assert rep["hard_flips"][0]["kind"] == "choice"
    assert rep["gate"]["passed"] is False

    # 2. Borderline choice flip (margin 0.001 < 0.02, diff=0.001) allowed under max_borderline_flips=1
    ref_b = [_sample_ref_case("c2", "q1", {"type": "choice", "choice": "A", "probabilities": {"A": 0.5005, "B": 0.4995}})]
    cand_b = [_cand_resp("q1", {"type": "choice", "choice": "B", "probabilities": {"A": 0.4995, "B": 0.5005}})]
    rep_b = da.evaluate_comparison(ref_b, cand_b, max_hard_flips=0, max_borderline_flips=1)
    assert len(rep_b["hard_flips"]) == 0
    assert len(rep_b["borderline_flips"]) == 1
    assert rep_b["borderline_coverage"]["count"] == 1
    assert rep_b["gate"]["passed"] is True

    # 3. Hard noul flip fails gate
    ref_n = [_sample_ref_case("c3", "n1", {"type": "noul", "noul": 0.65})]
    cand_n = [_cand_resp("n1", {"type": "noul", "noul": 0.45})]
    rep_n = da.evaluate_comparison(ref_n, cand_n)
    assert len(rep_n["hard_flips"]) == 1 and rep_n["hard_flips"][0]["kind"] == "noul"
    assert rep_n["gate"]["passed"] is False

    # 4. Score argmax flip + score round flip both detected
    ref_s = [_sample_ref_case(
        "c4",
        "s1",
        {"type": "score", "score": 1.25, "probabilities": {"0": 0.15, "1": 0.55, "2": 0.30}},
    )]
    cand_s = [_cand_resp(
        "s1",
        {"type": "score", "score": 1.60, "probabilities": {"0": 0.15, "1": 0.30, "2": 0.55}},
    )]
    rep_s = da.evaluate_comparison(ref_s, cand_s)
    kinds = {f["kind"] for f in rep_s["hard_flips"]}
    assert kinds == {"score_argmax", "score_round"}
    assert rep_s["gate"]["passed"] is False

    # 5. MAE threshold fails even with 0 flips
    ref_m = [_sample_ref_case("c5", "n1", {"type": "noul", "noul": 0.90})]
    cand_m = [_cand_resp("n1", {"type": "noul", "noul": 0.89})]
    rep_m = da.evaluate_comparison(ref_m, cand_m, max_prob_mae=0.002)
    assert len(rep_m["flips"]) == 0
    assert rep_m["gate"]["passed"] is False

    # 6. Isolated score_argmax-only flip, score_round-only flip, borderline cap, and <0 disable
    ref_sa = [_sample_ref_case("c6", "s1", {"type": "score", "score": 1.05, "probabilities": {"0": 0.1, "1": 0.55, "2": 0.35}})]
    cand_sa = [_cand_resp("s1", {"type": "score", "score": 1.15, "probabilities": {"0": 0.1, "1": 0.40, "2": 0.50}})]
    rep_sa = da.evaluate_comparison(ref_sa, cand_sa)
    assert [f["kind"] for f in rep_sa["hard_flips"]] == ["score_argmax"]
    assert rep_sa["gate"]["passed"] is False

    ref_sr = [_sample_ref_case("c7", "s1", {"type": "score", "score": 1.40, "probabilities": {"0": 0.1, "1": 0.60, "2": 0.30}})]
    cand_sr = [_cand_resp("s1", {"type": "score", "score": 1.60, "probabilities": {"0": 0.1, "1": 0.50, "2": 0.40}})]
    rep_sr = da.evaluate_comparison(ref_sr, cand_sr)
    assert [f["kind"] for f in rep_sr["hard_flips"]] == ["score_round"]
    assert rep_sr["gate"]["passed"] is False

    # Borderline flip limit enforced when > max_borderline_flips, and disabled when < 0
    rep_bf_fail = da.evaluate_comparison(ref_b, cand_b, max_hard_flips=0, max_borderline_flips=0)
    assert rep_bf_fail["gate"]["passed"] is False
    rep_bf_off = da.evaluate_comparison(ref_b, cand_b, max_hard_flips=0, max_borderline_flips=-1)
    assert rep_bf_off["gate"]["passed"] is True

    # 7. Choice fallback when 'choice' omitted + missing option key fails gate cleanly instead of KeyError
    ref_om = [_sample_ref_case("c8", "q1", {"type": "choice", "probabilities": {"A": 0.8, "B": 0.2}})]
    cand_om = [_cand_resp("q1", {"type": "choice", "probabilities": {"A": 0.8, "B": 0.2}})]
    assert da.evaluate_comparison(ref_om, cand_om)["gate"]["passed"] is True

    cand_missing = [_cand_resp("q1", {"type": "choice", "choice": "A", "probabilities": {"A": 0.8}})]
    rep_missing = da.evaluate_comparison(ref_om, cand_missing)
    assert len(rep_missing["missing_errors"]) == 1
    assert rep_missing["gate"]["passed"] is False


def test_evaluate_comparison_strict_schema_validation():
    ref_choice = [_sample_ref_case("c1", "q1", {"type": "choice", "choice": "A", "probabilities": {"A": 0.6, "B": 0.4}})]

    # 1. Choice mutated to score must fail
    rep_type = da.evaluate_comparison(
        ref_choice,
        [_cand_resp("q1", {"type": "score", "score": 0.4, "probabilities": {"A": 0.6, "B": 0.4}})],
    )
    assert rep_type["gate"]["passed"] is False
    assert "type mismatch" in rep_type["schema_errors"][0]["reason"]

    # 2. Choice preserving A/B (0.6/0.4) plus extra option C=0.9 must fail
    rep_extra_opt = da.evaluate_comparison(
        ref_choice,
        [_cand_resp("q1", {"type": "choice", "choice": "A", "probabilities": {"A": 0.6, "B": 0.4, "C": 0.9}})],
    )
    assert rep_extra_opt["gate"]["passed"] is False
    assert "unexpected extra choice probability keys" in rep_extra_opt["schema_errors"][0]["reason"]

    # 3. Score extra probability key must fail
    ref_score = [_sample_ref_case("c2", "s1", {"type": "score", "score": 0.5, "probabilities": {"0": 0.5, "1": 0.5}})]
    rep_extra_score = da.evaluate_comparison(
        ref_score,
        [_cand_resp("s1", {"type": "score", "score": 0.5, "probabilities": {"0": 0.5, "1": 0.5, "2": 0.0}})],
    )
    assert rep_extra_score["gate"]["passed"] is False
    assert "unexpected extra score probability keys" in rep_extra_score["schema_errors"][0]["reason"]

    # 4. Unexpected extra question ID must fail
    rep_extra_qid = da.evaluate_comparison(
        ref_choice,
        [(0.1, {
            "answers": {
                "q1": {"type": "choice", "choice": "A", "probabilities": {"A": 0.6, "B": 0.4}},
                "q_extra": {"type": "noul", "noul": 0.9},
            },
            "usage": {"input_tokens": 200},
        })],
    )
    assert rep_extra_qid["gate"]["passed"] is False
    assert "unexpected extra question" in rep_extra_qid["schema_errors"][0]["reason"]

    # 5. NaN, inf, bool, and non-dict probabilities fail cleanly without crashing
    for bad_val in (float("nan"), float("inf"), True):
        rep_bad_num = da.evaluate_comparison(
            ref_choice,
            [_cand_resp("q1", {"type": "choice", "choice": "A", "probabilities": {"A": bad_val, "B": 0.4}})],
        )
        assert rep_bad_num["gate"]["passed"] is False
        assert rep_bad_num["schema_errors"]

    rep_non_dict = da.evaluate_comparison(
        ref_choice,
        [_cand_resp("q1", {"type": "choice", "choice": "A", "probabilities": [0.6, 0.4]})],
    )
    assert rep_non_dict["gate"]["passed"] is False
    assert rep_non_dict["schema_errors"]

    # 6. Dropping choice when ref had choice, or invalid choice key, fails
    rep_drop_choice = da.evaluate_comparison(
        ref_choice,
        [_cand_resp("q1", {"type": "choice", "probabilities": {"A": 0.6, "B": 0.4}})],
    )
    assert rep_drop_choice["gate"]["passed"] is False
    assert "missing candidate choice" in rep_drop_choice["schema_errors"][0]["reason"]

    rep_bad_choice = da.evaluate_comparison(
        ref_choice,
        [_cand_resp("q1", {"type": "choice", "choice": "Z", "probabilities": {"A": 0.6, "B": 0.4}})],
    )
    assert rep_bad_choice["gate"]["passed"] is False
    assert "invalid candidate choice" in rep_bad_choice["schema_errors"][0]["reason"]

    # 7. Dropping score when ref had score fails
    rep_drop_score = da.evaluate_comparison(
        ref_score,
        [_cand_resp("s1", {"type": "score", "probabilities": {"0": 0.5, "1": 0.5}})],
    )
    assert rep_drop_score["gate"]["passed"] is False
    assert "missing candidate score value" in rep_drop_score["schema_errors"][0]["reason"]


def _mock_answers_for_body(req: dict) -> dict:
    answers = {}
    for qid, q in req["questions"].items():
        if q["type"] == "choice":
            opts = list(q["criteria"].keys())
            probs = {k: (0.8 if i == 0 else 0.2 / max(1, len(opts) - 1)) for i, k in enumerate(opts)}
            answers[qid] = {"type": "choice", "choice": opts[0], "probabilities": probs, "confidence": 0.8}
        elif q["type"] == "noul":
            answers[qid] = {"type": "noul", "noul": 0.9}
        else:
            lvls = q["criteria"]
            probs = {str(i): (0.7 if i == 0 else 0.3 / max(1, len(lvls) - 1)) for i in range(len(lvls))}
            answers[qid] = {"type": "score", "score": 0.4, "probabilities": probs}
    return answers


def test_evaluate_mixed_batch_coverage_and_schema():
    suite = da.build_suite()
    selected = [
        next(c for c in suite if c["name"] == name)
        for name in ("short_400_en_4opt_start", "med_1300_en_20opt_mid", "long_3300_zh_8opt_end", "max_7000_en_4opt_start")
    ]
    tok_map = [400, 1300, 3300, 7000]
    serial = [
        (0.1, {"answers": _mock_answers_for_body(c["body"]), "usage": {"input_tokens": tk}}, {"batch_size": 1, "batch_id": i + 1})
        for i, (c, tk) in enumerate(zip(selected, tok_map, strict=True))
    ]

    # 1. Missing batch headers -> uncovered
    batched_no_hdr = [
        (0.1, {"answers": _mock_answers_for_body(c["body"]), "usage": {"input_tokens": tk}}, {})
        for c, tk in zip(selected, tok_map, strict=True)
    ]
    rep_no_hdr = da.evaluate_mixed_batch(selected, serial, batched_no_hdr)
    assert rep_no_hdr["status"] == "uncovered"
    assert rep_no_hdr["covered"] is False

    # 2. All batch_size=1 -> uncovered
    batched_serial = [
        (0.1, {"answers": _mock_answers_for_body(c["body"]), "usage": {"input_tokens": tk}}, {"batch_size": 1, "batch_id": 10 + i})
        for i, (c, tk) in enumerate(zip(selected, tok_map, strict=True))
    ]
    rep_serial = da.evaluate_mixed_batch(selected, serial, batched_serial)
    assert rep_serial["status"] == "uncovered"
    assert rep_serial["covered"] is False

    # 3. Co-batched identical lengths (batch_size=2, same batch_id, identical input_tokens) -> uncovered
    same_tok_serial = [
        (0.1, {"answers": _mock_answers_for_body(c["body"]), "usage": {"input_tokens": 400}}, {"batch_size": 1, "batch_id": i + 1})
        for i, c in enumerate(selected)
    ]
    batched_same_len = [
        (0.1, {"answers": _mock_answers_for_body(selected[0]["body"]), "usage": {"input_tokens": 400}}, {"batch_size": 2, "batch_id": 10}),
        (0.1, {"answers": _mock_answers_for_body(selected[1]["body"]), "usage": {"input_tokens": 400}}, {"batch_size": 2, "batch_id": 10}),
        (0.1, {"answers": _mock_answers_for_body(selected[2]["body"]), "usage": {"input_tokens": 400}}, {"batch_size": 1, "batch_id": 11}),
        (0.1, {"answers": _mock_answers_for_body(selected[3]["body"]), "usage": {"input_tokens": 400}}, {"batch_size": 1, "batch_id": 12}),
    ]
    rep_same_len = da.evaluate_mixed_batch(selected, same_tok_serial, batched_same_len)
    assert rep_same_len["status"] == "uncovered"
    assert rep_same_len["covered"] is False

    # 4. Co-batched mixed lengths (batch_size=2, same batch_id, distinct input_tokens) -> pass
    batched_mixed = [
        (0.1, {"answers": _mock_answers_for_body(selected[0]["body"]), "usage": {"input_tokens": 400}}, {"batch_size": 2, "batch_id": 11}),
        (0.1, {"answers": _mock_answers_for_body(selected[1]["body"]), "usage": {"input_tokens": 1300}}, {"batch_size": 2, "batch_id": 11}),
        (0.1, {"answers": _mock_answers_for_body(selected[2]["body"]), "usage": {"input_tokens": 3300}}, {"batch_size": 1, "batch_id": 12}),
        (0.1, {"answers": _mock_answers_for_body(selected[3]["body"]), "usage": {"input_tokens": 7000}}, {"batch_size": 1, "batch_id": 13}),
    ]
    rep_ok = da.evaluate_mixed_batch(selected, serial, batched_mixed)
    assert rep_ok["status"] == "pass"
    assert rep_ok["passed"] is True
    assert rep_ok["covered"] is True
    assert len(rep_ok["mixed_batches"]) == 1

    # 5. Co-batched mixed lengths with structural / drift regressions -> fail
    def _clone_mixed():
        return [
            (lat, json.loads(json.dumps(resp)), dict(meta))
            for lat, resp, meta in batched_mixed
        ]

    # 5a. Mutated type
    bad_type = _clone_mixed()
    bad_type[0][1]["answers"]["pick"]["type"] = "score"
    assert da.evaluate_mixed_batch(selected, serial, bad_type)["status"] == "fail"

    # 5b. Extra option key
    bad_opt = _clone_mixed()
    bad_opt[0][1]["answers"]["pick"]["probabilities"]["EXTRA"] = 0.9
    assert da.evaluate_mixed_batch(selected, serial, bad_opt)["status"] == "fail"

    # 5c. Extra question ID
    bad_qid = _clone_mixed()
    bad_qid[0][1]["answers"]["extra_q"] = {"type": "noul", "noul": 0.5}
    assert da.evaluate_mixed_batch(selected, serial, bad_qid)["status"] == "fail"

    # 5d. Corrupted score scalar
    bad_score = _clone_mixed()
    bad_score[0][1]["answers"]["score_hit"]["score"] = 0.9
    rep_score = da.evaluate_mixed_batch(selected, serial, bad_score)
    assert rep_score["status"] == "fail"
    assert rep_score["max_score_diff_detail"]["case"] == selected[0]["name"]
    assert rep_score["max_score_diff_detail"]["qid"] == "score_hit"

    # 5e. Probability drift > 5e-3 fails default, while small GPU-like FP diff (2.5e-3) passes default 5e-3
    small_fp_drift = _clone_mixed()
    small_fp_drift[0][1]["answers"]["has_tls"]["noul"] = 0.8975
    assert da.evaluate_mixed_batch(selected, serial, small_fp_drift)["status"] == "pass"
    assert da.evaluate_mixed_batch(selected, serial, small_fp_drift, max_prob_diff=1e-3)["status"] == "fail"

    bad_drift = _clone_mixed()
    bad_drift[0][1]["answers"]["has_tls"]["noul"] = 0.85
    rep_drift = da.evaluate_mixed_batch(selected, serial, bad_drift)
    assert rep_drift["status"] == "fail"
    assert rep_drift["max_prob_diff_detail"]["case"] == selected[0]["name"]
    assert rep_drift["max_prob_diff_detail"]["qid"] == "has_tls"
    assert rep_drift["max_prob_diff_detail"]["key"] is None


def test_cli_record_and_compare_check_exit_codes(tmp_path: Path):
    flip_target = {"enabled": False}
    batch_headers = {"mode": "none", "seq": 0}
    hdr_lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            if self.path == "/version":
                body = json.dumps({"version": "mock-1.0"}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            req = json.loads(self.rfile.read(length).decode("utf-8"))
            answers = {}
            for qid, q in req["questions"].items():
                if q["type"] == "choice":
                    opts = list(q["criteria"].keys())
                    winner = opts[1] if (flip_target["enabled"] and qid == "pick") else opts[0]
                    probs = {k: (0.8 if k == winner else 0.2 / max(1, len(opts) - 1)) for k in opts}
                    answers[qid] = {"type": "choice", "choice": winner, "probabilities": probs, "confidence": 0.8}
                elif q["type"] == "noul":
                    answers[qid] = {"type": "noul", "noul": 0.9}
                else:
                    lvls = q["criteria"]
                    probs = {str(i): (0.7 if i == 0 else 0.3 / max(1, len(lvls) - 1)) for i in range(len(lvls))}
                    answers[qid] = {"type": "score", "score": 0.4, "probabilities": probs}
            resp = json.dumps({"answers": answers, "usage": {"input_tokens": 500, "output_tokens": 0}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            if batch_headers["mode"] == "serial":
                with hdr_lock:
                    batch_headers["seq"] += 1
                    bid = batch_headers["seq"]
                self.send_header("X-Decide-Batch-Size", "1")
                self.send_header("X-Decide-Batch-Id", str(bid))
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)

    srv_http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = srv_http.server_address[1]
    t = threading.Thread(target=srv_http.serve_forever, daemon=True)
    t.start()
    try:
        ref_file = tmp_path / "ref.json"
        out_json = tmp_path / "report.json"
        script = str(TOOLS_DIR / "diff_accuracy.py")
        url = f"http://127.0.0.1:{port}"

        rec = subprocess.run(
            [sys.executable, script, "--record", str(ref_file), "--url", url, "--key", "test", "--max-tier-tokens", "500"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert rec.returncode == 0, rec.stderr + rec.stdout
        assert ref_file.exists()

        ok = subprocess.run(
            [sys.executable, script, "--compare", str(ref_file), "--url", url, "--key", "test", "--check", "--json-out", str(out_json)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert ok.returncode == 0, ok.stdout + ok.stderr
        assert json.loads(out_json.read_text(encoding="utf-8"))["gate"]["passed"] is True

        flip_target["enabled"] = True
        fail = subprocess.run(
            [sys.executable, script, "--compare", str(ref_file), "--url", url, "--key", "test", "-c", "2", "--check"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert fail.returncode == 1
        assert "Regression Gate: FAIL" in fail.stdout

        # Mixed-batch without batch headers exits 3 (NOT COVERED)
        flip_target["enabled"] = False
        mb_no_hdr = subprocess.run(
            [sys.executable, script, "--mixed-batch", "--url", url, "--key", "test"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert mb_no_hdr.returncode == 3, mb_no_hdr.stdout + mb_no_hdr.stderr
        assert "NOT COVERED" in mb_no_hdr.stderr

        # Mixed-batch with serial batch_size=1 headers also exits 3 (NOT COVERED)
        batch_headers["mode"] = "serial"
        mb_serial = subprocess.run(
            [sys.executable, script, "--mixed-batch", "--url", url, "--key", "test"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert mb_serial.returncode == 3, mb_serial.stdout + mb_serial.stderr
        assert "NOT COVERED" in mb_serial.stderr
    finally:
        srv_http.shutdown()
        srv_http.server_close()
        t.join(timeout=3.0)


def test_cli_mixed_batch_against_real_server_app(monkeypatch: pytest.MonkeyPatch):
    pkg = types.ModuleType("hindsight_decision")
    schema_mod = types.ModuleType("hindsight_decision.schema")
    endpoint_mod = types.ModuleType("hindsight_decision.endpoint")
    identity_mod = types.ModuleType("hindsight_decision.identity")
    typed_mod = types.ModuleType("hindsight_decision.typed")
    typed_render_mod = types.ModuleType("hindsight_decision.typed_render")

    class ContextOverflow(Exception):
        pass

    class _Identity:
        def headers(self) -> dict[str, str]:
            return {"X-Decision-Server": "test"}

    async def answer_body(raw_request, handler):
        body = await raw_request.json()
        return await handler(body)

    class _Rendered:
        def __init__(self, tok_count: int, q_count: int) -> None:
            self._tokens = tok_count
            self.questions = [types.SimpleNamespace(slot=i) for i in range(q_count)]
            self.ids = [1] * tok_count

        def tokens(self) -> int:
            return self._tokens

    def render_request(_tokenizer, request, _settings):
        toks = max(1, len(json.dumps(request.get("state", ""), ensure_ascii=False)) // 4)
        return _Rendered(toks, max(1, len(request.get("questions") or {})))

    def response(request, rendered, _rows, _ids):
        return {
            "answers": _mock_answers_for_body(request),
            "usage": {"input_tokens": rendered.tokens(), "output_tokens": 0},
        }

    schema_mod.ContextOverflow = ContextOverflow
    endpoint_mod.answer_body = answer_body
    identity_mod.identity = lambda _m, _f, _r: _Identity()
    identity_mod.models_with_identity = lambda raw, _s: raw
    typed_mod.response = response
    typed_render_mod.render_request = render_request
    pkg.schema = schema_mod
    pkg.endpoint = endpoint_mod
    pkg.identity = identity_mod
    pkg.typed = typed_mod
    pkg.typed_render = typed_render_mod

    monkeypatch.setitem(sys.modules, "hindsight_decision", pkg)
    monkeypatch.setitem(sys.modules, "hindsight_decision.schema", schema_mod)
    monkeypatch.setitem(sys.modules, "hindsight_decision.endpoint", endpoint_mod)
    monkeypatch.setitem(sys.modules, "hindsight_decision.identity", identity_mod)
    monkeypatch.setitem(sys.modules, "hindsight_decision.typed", typed_mod)
    monkeypatch.setitem(sys.modules, "hindsight_decision.typed_render", typed_render_mod)

    script = str(TOOLS_DIR / "diff_accuracy.py")

    def _run_with_app(decider, batch_wait_ms: int) -> subprocess.CompletedProcess[str]:
        app = srv.create_app(
            decider,
            "hs-decide-1-flash",
            None,
            ["test"],
            batch_wait_ms=batch_wait_ms,
            use_color=False,
            max_queue_size=16,
            max_queue_per_key=16,
            queue_timeout_s=10.0,
        )
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]

        config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
        uv_srv = uvicorn.Server(config)
        t = threading.Thread(target=uv_srv.run, daemon=True)
        t.start()
        try:
            import urllib.request

            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                try:
                    req = urllib.request.Request(
                        f"http://127.0.0.1:{port}/health",
                        headers={"Authorization": "Bearer test"},
                    )
                    with urllib.request.urlopen(req, timeout=1.0) as r:
                        if r.status == 200:
                            break
                except Exception:
                    time.sleep(0.02)
            else:
                raise RuntimeError("Uvicorn server failed to become ready")

            return subprocess.run(
                [sys.executable, script, "--mixed-batch", "--url", f"http://127.0.0.1:{port}", "--key", "test"],
                capture_output=True,
                text=True,
                check=False,
            )
        finally:
            uv_srv.should_exit = True
            t.join(timeout=5.0)

    class _LocalDecider:
        def __init__(self, max_batch_size: int) -> None:
            self.folder = Path(".")
            self.settings = types.SimpleNamespace(max_request_tokens=32768)
            self.tokenizer = None
            self.ids = [1, 2]
            self.ctx_size = 32768
            self.max_batch_tokens = 32768
            self.max_batch_size = max_batch_size
            self.max_outputs = 256
            self.last_batch_fell_back = False

        def engine_version(self) -> str:
            return "fake-llamacpp"

        def close(self) -> None:
            pass

        def eval_batch(self, rendered_list):
            time.sleep(0.02)
            return [[[0.0, 1.0] for _ in r.questions] for r in rendered_list]

    # 1. max_batch_size=1 -> exits 3 (NOT COVERED)
    res_serial = _run_with_app(_LocalDecider(max_batch_size=1), batch_wait_ms=0)
    assert res_serial.returncode == 3, res_serial.stdout + res_serial.stderr
    assert "NOT COVERED" in res_serial.stderr

    # 2. max_batch_size=4 with batch_wait_ms=200 -> forms real mixed-length batch, exits 0
    res_batched = _run_with_app(_LocalDecider(max_batch_size=4), batch_wait_ms=200)
    assert res_batched.returncode == 0, res_batched.stdout + res_batched.stderr
    assert "mixed_batches=1/" in res_batched.stdout

    # 3. max_batch_size=4 with LlamaCppDecider.eval_batch serial fallback -> exits 3 (NOT COVERED)
    fb_decider = object.__new__(srv.LlamaCppDecider)
    fb_decider.folder = Path(".")
    fb_decider.settings = types.SimpleNamespace(max_request_tokens=32768)
    fb_decider.tokenizer = None
    fb_decider.ids = [1, 2]
    fb_decider.ctx_size = 32768
    fb_decider.max_batch_tokens = 32768
    fb_decider.max_batch_size = 4
    fb_decider.max_outputs = 256
    fb_decider.device_label = "cpu"
    fb_decider.last_batch_fell_back = False
    fallback_attempts = 0

    def _fb_eval_once(rendered_list):
        nonlocal fallback_attempts
        if len(rendered_list) > 1:
            fallback_attempts += 1
            raise RuntimeError("simulated multi-seq eval failure")
        return [[[0.0, 1.0] for _ in rendered_list[0].questions]]

    fb_decider._eval_once = _fb_eval_once
    res_fb = _run_with_app(fb_decider, batch_wait_ms=200)
    assert fallback_attempts >= 1
    assert res_fb.returncode == 3, res_fb.stdout + res_fb.stderr
    assert "NOT COVERED" in res_fb.stderr
