import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import diff_accuracy as da  # noqa: E402


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


def test_cli_record_and_compare_check_exit_codes(tmp_path: Path):
    flip_target = {"enabled": False}

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
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
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

        # Mixed-batch consistency mode passes when consistent
        flip_target["enabled"] = False
        mb_ok = subprocess.run(
            [sys.executable, script, "--mixed-batch", "--url", url, "--key", "test"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert mb_ok.returncode == 0, mb_ok.stdout + mb_ok.stderr
    finally:
        srv.shutdown()
        srv.server_close()
        t.join(timeout=3.0)
