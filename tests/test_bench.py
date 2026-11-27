import http.client
import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import bench  # noqa: E402


def test_parse_timing_headers():
    q, e = bench.parse_timing_headers({
        "X-Decide-Queue-Ms": "12.50",
        "X-Decide-Eval-Ms": "148.25",
    })
    assert q == 12.5 and e == 148.25

    q2, e2 = bench.parse_timing_headers({
        "server-timing": "queue;desc=q;dur=3.40, eval;dur=99.10",
    })
    assert q2 == 3.4 and e2 == 99.1

    # Multiple Server-Timing header lines + fallback when only one X-Decide header is set
    msg = http.client.HTTPMessage()
    msg.add_header("X-Decide-Queue-Ms", "-1")
    msg.add_header("Server-Timing", "proxy;dur=1.2")
    msg.add_header("Server-Timing", "queue;dur=4.5, eval;dur=88.0")
    q_m, e_m = bench.parse_timing_headers(msg)
    assert q_m == 4.5 and e_m == 88.0

    # Malformed / NaN / inf values ignored cleanly
    q_bad, e_bad = bench.parse_timing_headers({
        "X-Decide-Queue-Ms": "nan",
        "X-Decide-Eval-Ms": "inf",
        "Server-Timing": "queue;dur=-5, eval;dur=abc",
    })
    assert q_bad is None and e_bad is None

    q3, e3 = bench.parse_timing_headers({"X-Other": "1"})
    assert q3 is None and e3 is None


def test_parse_keys_and_assign_keys():
    # Env key does NOT override or shift --keys
    assert bench.parse_keys(None, "env_k", "k1, k2") == ["k1", "k2"]
    # Explicit CLI --key merges ahead of --keys without duplicating
    assert bench.parse_keys("k0", "env_k", "k1, k2") == ["k0", "k1", "k2"]
    assert bench.parse_keys("k2", "env_k", "k1, k2") == ["k1", "k2"]
    assert bench.parse_keys(None, "env_k", None) == ["env_k"]

    l1, s1 = bench.assign_keys(["k0"], 2, 2)
    assert l1 == [(0, "k0"), (0, "k0")]
    assert s1 == [(0, "k0"), (0, "k0")]

    l2, s2 = bench.assign_keys(["k0", "k1"], 1, 1)
    assert l2 == [(0, "k0")]
    assert s2 == [(1, "k1")]

    l3, s3 = bench.assign_keys(["k0", "k1", "k2"], 2, 1)
    assert l3 == [(0, "k0"), (1, "k1")]
    assert s3 == [(2, "k2")]

    l4, s4 = bench.assign_keys(["k0", "k1"], 3, 2)
    assert l4 == [(0, "k0"), (1, "k1"), (0, "k0")]
    assert s4 == [(1, "k1"), (0, "k0")]


def test_bench_sweep_and_mixed_cli():
    seen_keys: list[str] = []
    lock = threading.Lock()
    rate_limit_short = {"enabled": False}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            if self.path == "/version":
                body = b'{"version":"mock-bench"}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        def do_POST(self):
            auth = self.headers.get("Authorization", "")
            length = int(self.headers.get("Content-Length", "0"))
            req = json.loads(self.rfile.read(length).decode("utf-8"))
            n_docs = len(req["state"]["documents"])
            with lock:
                seen_keys.append(auth)

            if rate_limit_short["enabled"] and n_docs < 20:
                body = b'{"error":"too_many_requests"}'
                self.send_response(429)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            time.sleep(0.03 if n_docs >= 20 else 0.005)
            toks = 7000 if n_docs >= 20 else 400
            q_ms = 5.0 if n_docs >= 20 else 12.5
            e_ms = 30.0 if n_docs >= 20 else 5.0
            body = json.dumps({"answers": {}, "usage": {"input_tokens": toks, "output_tokens": 0}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("X-Decide-Queue-Ms", str(q_ms))
            self.send_header("X-Decide-Eval-Ms", str(e_ms))
            self.send_header("Server-Timing", f"queue;dur={q_ms}, eval;dur={e_ms}")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        script = str(TOOLS_DIR / "bench.py")
        url = f"http://127.0.0.1:{port}"

        # 1. Normal sweep JSON
        sweep = subprocess.run(
            [sys.executable, script, "--url", url, "--key", "k0", "--sizes", "400", "-n", "3", "-c", "2", "--warmup", "1", "--json"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert sweep.returncode == 0, sweep.stderr + sweep.stdout
        sweep_data = json.loads(sweep.stdout)
        assert len(sweep_data) == 1
        assert sweep_data[0]["ok"] == 3
        assert sweep_data[0]["q_p95"] == 12.5

        # 2. Mixed mode with two keys (verify DECIDE_API_KEY env does NOT leak into --keys)
        seen_keys.clear()
        env_with_key = {**os.environ, "DECIDE_API_KEY": "env_leaked_key"}
        mixed = subprocess.run(
            [
                sys.executable, script,
                "--url", url,
                "--mixed",
                "--keys", "k0,k1",
                "--short-size", "400",
                "--long-size", "7000",
                "--short-concurrency", "1",
                "--long-concurrency", "1",
                "--short-interval-ms", "15",
                "--duration", "0.4",
                "--warmup", "1",
                "--json",
            ],
            env=env_with_key,
            capture_output=True,
            text=True,
            check=False,
        )
        assert mixed.returncode == 0, mixed.stderr + mixed.stdout
        mdata = json.loads(mixed.stdout)
        assert mdata["mode"] == "mixed"
        assert mdata["long_key_slots"] == [0]
        assert mdata["short_key_slots"] == [1]
        assert mdata["short"]["ok"] > 0
        assert mdata["long"]["ok"] > 0
        assert mdata["short"]["queue_ms"]["p95"] == 12.5
        assert mdata["aggregate"]["total_tok_s"] > 0
        assert set(seen_keys) == {"Bearer k0", "Bearer k1"}

        # 3. Text-mode output works cleanly too
        mixed_txt = subprocess.run(
            [
                sys.executable, script,
                "--url", url,
                "--mixed",
                "--keys", "k0,k1",
                "--short-size", "400",
                "--long-size", "7000",
                "--duration", "0.25",
                "--warmup", "0",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert mixed_txt.returncode == 0, mixed_txt.stderr + mixed_txt.stdout
        assert "short" in mixed_txt.stdout and "long" in mixed_txt.stdout

        # 4. Mixed mode tracks 429 errors cleanly without crashing, and invalid -n 0 exits cleanly
        rate_limit_short["enabled"] = True
        mixed_err = subprocess.run(
            [
                sys.executable, script,
                "--url", url,
                "--mixed",
                "--keys", "k0,k1",
                "--short-size", "400",
                "--long-size", "7000",
                "--duration", "0.3",
                "--warmup", "0",
                "--json",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert mixed_err.returncode == 0, mixed_err.stderr + mixed_err.stdout
        medata = json.loads(mixed_err.stdout)
        assert medata["short"]["fail"] > 0
        assert medata["short"]["errors_by_code"]["HTTP 429"] == medata["short"]["fail"]

        bad_n = subprocess.run(
            [sys.executable, script, "--url", url, "--key", "k0", "--sizes", "400", "-n", "0"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert bad_n.returncode != 0 and "Traceback" not in bad_n.stderr
    finally:
        srv.shutdown()
        srv.server_close()
        t.join(timeout=3.0)
