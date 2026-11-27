import asyncio
import http.client
import json
import logging
import socket
import sys
import threading
import time
import types
from contextlib import contextmanager
from pathlib import Path

import pytest
import uvicorn

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import server as srv  # noqa: E402


class _FakeRendered:
    def __init__(self, tag: str, tokens_count: int = 100, questions: int = 1) -> None:
        self.tag = tag
        self._tokens = tokens_count
        self.questions = [types.SimpleNamespace(slot=i) for i in range(questions)]
        self.ids = list(range(tokens_count))

    def tokens(self) -> int:
        return self._tokens


def _install_fake_hindsight_decision(monkeypatch: pytest.MonkeyPatch, render_calls: list[str]) -> None:
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
        try:
            return await handler(body)
        except ContextOverflow as exc:
            from fastapi.responses import JSONResponse

            return JSONResponse(
                {"detail": {"error_type": "max_tokens_exceeded", "message": str(exc)}},
                status_code=400,
            )

    def render_request(_tokenizer, request, _settings):
        state = request.get("state") or {}
        tag = state.get("tag", "req") if isinstance(state, dict) else "req"
        tok = int(state.get("tokens", 120)) if isinstance(state, dict) else 120
        q_count = max(1, len(request.get("questions") or {"q": {}}))
        render_calls.append(tag)
        return _FakeRendered(tag, tokens_count=tok, questions=q_count)

    def response(request, rendered, _rows, _ids):
        answers = {}
        for qid, q in (request.get("questions") or {"q": {"type": "noul"}}).items():
            qt = q.get("type", "noul")
            if qt == "noul":
                answers[qid] = {"type": "noul", "noul": 0.9}
            elif qt == "choice":
                opts = list((q.get("criteria") or {"a": None, "b": None}).keys())
                answers[qid] = {
                    "type": "choice",
                    "choice": opts[0],
                    "probabilities": {k: (0.9 if i == 0 else 0.1 / max(1, len(opts) - 1)) for i, k in enumerate(opts)},
                    "confidence": 0.9,
                }
            else:
                answers[qid] = {"type": "score", "score": 1.0, "probabilities": {"0": 0.2, "1": 0.8}}
        return {
            "answers": answers,
            "usage": {"input_tokens": rendered.tokens(), "output_tokens": 0},
            "tag": rendered.tag,
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


class _FakeDecider:
    def __init__(self) -> None:
        self.folder = Path(".")
        self.settings = types.SimpleNamespace(max_request_tokens=16384)
        self.tokenizer = None
        self.ids = [1, 2]
        self.ctx_size = 16384
        self.max_batch_tokens = 16384
        self.max_batch_size = 1
        self.max_outputs = 256
        self.eval_calls: list[str] = []
        self.started_events: dict[str, threading.Event] = {}
        self.release_events: dict[str, threading.Event] = {}
        self.min_eval_s: float = 0.0
        self._lock = threading.Lock()

    def arm(self, tag: str) -> tuple[threading.Event, threading.Event]:
        started = threading.Event()
        release = threading.Event()
        with self._lock:
            self.started_events[tag] = started
            self.release_events[tag] = release
        return started, release

    def engine_version(self) -> str:
        return "fake-llamacpp"

    def close(self) -> None:
        pass

    def eval_batch(self, rendered_list):
        if self.min_eval_s > 0:
            time.sleep(self.min_eval_s)
        for r in rendered_list:
            with self._lock:
                self.eval_calls.append(r.tag)
                started = self.started_events.get(r.tag)
                release = self.release_events.get(r.tag)
            if started is not None:
                started.set()
            if release is not None:
                release.wait(timeout=10.0)
        return [[[0.0, 1.0] for _ in r.questions] for r in rendered_list]


@contextmanager
def _running_uvicorn(app, api_key: str):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    try:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            try:
                conn = http.client.HTTPConnection("127.0.0.1", port, timeout=1.0)
                conn.request("GET", "/health", headers={"Authorization": f"Bearer {api_key}"})
                resp = conn.getresponse()
                resp.read()
                conn.close()
                if resp.status == 200:
                    break
            except Exception:
                time.sleep(0.02)
        else:
            raise RuntimeError("Uvicorn test server failed to become ready")
        yield port
    finally:
        server.should_exit = True
        t.join(timeout=5.0)


def _post_systemone(port: int, key: str, tag: str, timeout: float = 5.0, tokens: int = 120):
    payload = json.dumps({
        "model": "hs-decide-1-flash",
        "state": {"tag": tag, "tokens": tokens},
        "questions": {"q": {"type": "noul", "instructions": "ok?"}},
    }).encode("utf-8")
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        conn.request(
            "POST",
            "/v1/systemone",
            body=payload,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )
        resp = conn.getresponse()
        body = resp.read().decode("utf-8")
        headers = {k.lower(): v for k, v in resp.getheaders()}
        ctype = headers.get("content-type", "")
        parsed = json.loads(body) if (body and "json" in ctype) else (body or None)
        return resp.status, headers, parsed
    finally:
        conn.close()


def _open_raw_post(port: int, key: str, tag: str, tokens: int = 120) -> socket.socket:
    body = json.dumps({
        "model": "hs-decide-1-flash",
        "state": {"tag": tag, "tokens": tokens},
        "questions": {"q": {"type": "noul", "instructions": "ok?"}},
    }).encode("utf-8")
    raw_req = (
        f"POST /v1/systemone HTTP/1.1\r\n"
        f"Host: 127.0.0.1:{port}\r\n"
        f"Authorization: Bearer {key}\r\n"
        f"Content-Type: application/json\r\n"
        f"Content-Length: {len(body)}\r\n"
        f"Connection: close\r\n\r\n"
    ).encode("ascii") + body
    sock = socket.create_connection(("127.0.0.1", port), timeout=3.0)
    sock.sendall(raw_req)
    return sock


def _wait_rendered(render_calls: list[str], tag: str, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while tag not in render_calls and time.monotonic() < deadline:
        time.sleep(0.01)
    assert tag in render_calls


def _make_item(
    loop: asyncio.AbstractEventLoop,
    tag: str,
    enqueued_at: float = 0.0,
    deadline: float | None = None,
    request: dict | None = None,
) -> srv.PendingItem:
    return srv.PendingItem(
        request=request if request is not None else {},
        rendered=_FakeRendered(tag),
        future=loop.create_future(),
        enqueued_at=enqueued_at,
        deadline=deadline,
        started_event=asyncio.Event(),
    )


# --- Unit tests for FairQueue ---


def test_fair_queue_limits_and_expiry():
    loop = asyncio.new_event_loop()
    try:
        fq = srv.FairQueue(max_size=3, max_per_key=2)
        i1 = _make_item(loop, "k0", enqueued_at=10.0, deadline=20.0)
        i2 = _make_item(loop, "k0", enqueued_at=10.0, deadline=25.0)
        fq.push("k0", i1, now=10.0)
        fq.push("k0", i2, now=10.0)

        # Per-key cap triggers 429 before global cap
        with pytest.raises(srv.QueueRejectedError) as exc_key:
            fq.push("k0", _make_item(loop, "k0", enqueued_at=10.0), now=10.0)
        assert exc_key.value.status_code == 429

        # Different key can push up to global cap (3)
        i3 = _make_item(loop, "k1", enqueued_at=10.0, deadline=30.0)
        fq.push("k1", i3, now=10.0)
        with pytest.raises(srv.QueueRejectedError) as exc_global:
            fq.push("k2", _make_item(loop, "k2", enqueued_at=10.0), now=10.0)
        assert exc_global.value.status_code == 503

        # Expiring i1 via prune frees both per-key and global capacity
        removed = fq.prune(now=21.0)
        assert removed == 1
        assert i1.cancelled is True
        assert isinstance(i1.future.exception(), srv.QueueTimeoutError)
        assert fq.total_pending() == 2
        assert fq.tenant_pending("k0") == 1

        # Round-robin pops k0 (i2) then k1 (i3)
        tid_a, item_a = fq.pop_next(now=21.0)
        assert (tid_a, item_a) == ("k0", i2)
        tid_b, item_b = fq.pop_next(now=21.0)
        assert (tid_b, item_b) == ("k1", i3)
        assert fq.pop_next(now=21.0) is None
    finally:
        loop.close()


def test_fair_queue_pop_next_skips_expired_and_remove_uses_identity():
    loop = asyncio.new_event_loop()
    try:
        fq = srv.FairQueue(max_size=10, max_per_key=10)
        exp = _make_item(loop, "exp", deadline=5.0, request={"x": 1})
        live1 = _make_item(loop, "live1", deadline=50.0, request={"x": 1})
        live2 = _make_item(loop, "live2", deadline=50.0, request={"x": 1})
        fq.push("k0", exp, now=0.0)
        fq.push("k0", live1, now=0.0)
        fq.push("k0", live2, now=0.0)

        # Remove live2 by identity even though live1 has equal payload
        assert fq.remove("k0", live2) is True
        assert fq.total_pending() == 2

        popped = fq.pop_next(now=6.0)
        assert popped is not None and popped[1] is live1
        assert isinstance(exp.future.exception(), srv.QueueTimeoutError)
        assert fq.total_pending() == 0
    finally:
        loop.close()


def test_negative_queue_limits_rejected(monkeypatch: pytest.MonkeyPatch):
    render_calls: list[str] = []
    _install_fake_hindsight_decision(monkeypatch, render_calls)
    decider = _FakeDecider()
    with pytest.raises(ValueError):
        srv.FairQueue(max_size=-1)
    with pytest.raises(ValueError):
        srv.FairQueue(max_per_key=-1)
    with pytest.raises(ValueError):
        srv.create_app(decider, "m", None, ["k0"], 0, max_queue_size=-1)
    with pytest.raises(ValueError):
        srv.create_app(decider, "m", None, ["k0"], 0, max_queue_per_key=-1)
    with pytest.raises(ValueError):
        srv.create_app(decider, "m", None, ["k0"], 0, queue_timeout_s=-0.5)


# --- Real Uvicorn HTTP integration tests ---


def test_timing_headers_and_429_503_limits(monkeypatch: pytest.MonkeyPatch):
    render_calls: list[str] = []
    _install_fake_hindsight_decision(monkeypatch, render_calls)
    decider = _FakeDecider()
    decider.min_eval_s = 0.01
    blocker_started, release_blocker = decider.arm("blocker")

    app = srv.create_app(
        decider,
        "hs-decide-1-flash",
        None,
        ["k0", "k1", "k2"],
        0,
        use_color=False,
        max_queue_size=2,
        max_queue_per_key=1,
        queue_timeout_s=10.0,
    )

    with _running_uvicorn(app, "k0") as port:
        results: dict[str, tuple] = {}
        threads = []
        try:
            def launch(name: str, key: str, tag: str):
                def _run():
                    results[name] = _post_systemone(port, key, tag)

                t = threading.Thread(target=_run)
                t.start()
                threads.append(t)

            # 1. Blocker enters GPU eval (not queued anymore)
            launch("blocker", "k0", "blocker")
            assert blocker_started.wait(timeout=3.0)

            # 2. One item per key fits in queue (global max=2, per-key max=1)
            launch("q_k0", "k0", "q_k0")
            _wait_rendered(render_calls, "q_k0")

            # Second queued item for k0 must fail immediately with 429 BEFORE render_request
            before_render_len = len(render_calls)
            st_429, hdr_429, body_429 = _post_systemone(port, "k0", "overflow_k0")
            assert st_429 == 429
            assert hdr_429.get("retry-after") == "2"
            assert body_429["error"] == "too_many_requests"
            assert len(render_calls) == before_render_len

            # k1 can still take the remaining global queue slot
            launch("q_k1", "k1", "q_k1")
            _wait_rendered(render_calls, "q_k1")

            # Fresh key k2 (0/1 per-key) hits global cap (2/2) -> 503
            st_503, hdr_503, body_503 = _post_systemone(port, "k2", "overflow_global")
            assert st_503 == 503
            assert hdr_503.get("retry-after") == "2"
            assert body_503["error"] == "server_busy"
        finally:
            release_blocker.set()
            for t in threads:
                t.join(timeout=5.0)

        batch_ids: set[str] = set()
        for name in ("blocker", "q_k0", "q_k1"):
            st, hdr, data = results[name]
            assert st == 200
            assert "x-decide-queue-ms" in hdr
            assert "x-decide-eval-ms" in hdr
            assert hdr.get("x-decide-batch-size") == "1"
            assert "x-decide-batch-id" in hdr and int(hdr["x-decide-batch-id"]) >= 1
            batch_ids.add(hdr["x-decide-batch-id"])
            assert "queue;dur=" in hdr["server-timing"]
            assert "eval;dur=" in hdr["server-timing"]
            assert data["tag"] == name
        assert len(batch_ids) == 3


def test_queue_timeout_evicts_pre_start_but_never_aborts_started(monkeypatch: pytest.MonkeyPatch):
    render_calls: list[str] = []
    _install_fake_hindsight_decision(monkeypatch, render_calls)
    decider = _FakeDecider()
    blocker_started, release_blocker = decider.arm("blocker")
    survivor_started, release_survivor = decider.arm("survivor")

    app = srv.create_app(
        decider,
        "hs-decide-1-flash",
        None,
        ["k0"],
        0,
        use_color=False,
        max_queue_size=4,
        max_queue_per_key=4,
        queue_timeout_s=0.35,
    )

    with _running_uvicorn(app, "k0") as port:
        results: dict[str, tuple] = {}

        def launch(tag: str):
            t = threading.Thread(target=lambda: results.__setitem__(tag, _post_systemone(port, "k0", tag)))
            t.start()
            return t

        t_blocker = launch("blocker")
        assert blocker_started.wait(timeout=3.0)

        # Case A: victim waits past queue_timeout_s=0.35s while blocker holds GPU -> 503 queue_timeout
        t_victim = launch("victim")
        t_victim.join(timeout=5.0)
        st_v, hdr_v, body_v = results["victim"]
        assert st_v == 503
        assert hdr_v.get("retry-after") == "5"
        assert body_v["error"] == "queue_timeout"

        # Case B: survivor enqueues, blocker releases immediately so survivor starts BEFORE 0.35s,
        # then survivor runs on GPU for 0.55s (> queue_timeout_s) and MUST still return 200 OK.
        t_survivor = launch("survivor")
        _wait_rendered(render_calls, "survivor")

        release_blocker.set()
        t_blocker.join(timeout=5.0)
        assert survivor_started.wait(timeout=3.0)
        time.sleep(0.55)
        release_survivor.set()
        t_survivor.join(timeout=5.0)

        st_s, _, data_s = results["survivor"]
        assert st_s == 200
        assert data_s["tag"] == "survivor"
        assert "victim" not in decider.eval_calls


def test_real_tcp_disconnect_evicts_queued_request_cleanly(monkeypatch: pytest.MonkeyPatch, capsys):
    render_calls: list[str] = []
    _install_fake_hindsight_decision(monkeypatch, render_calls)
    decider = _FakeDecider()
    blocker_started, release_blocker = decider.arm("blocker")

    # Capture uvicorn + access logs to ensure zero RuntimeError / ASGI errors on disconnect
    uvicorn_errors: list[str] = []

    class _ErrCap(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            if record.levelno >= logging.ERROR:
                uvicorn_errors.append(record.getMessage())

    err_handler = _ErrCap()
    logging.getLogger("uvicorn.error").addHandler(err_handler)

    app = srv.create_app(
        decider,
        "hs-decide-1-flash",
        None,
        ["k0"],
        0,
        use_color=False,
        max_queue_size=1,
        max_queue_per_key=1,
        queue_timeout_s=10.0,
    )

    try:
        with _running_uvicorn(app, "k0") as port:
            blocker_res = []
            t_blocker = threading.Thread(target=lambda: blocker_res.append(_post_systemone(port, "k0", "blocker")))
            t_blocker.start()
            assert blocker_started.wait(timeout=3.0)

            # Connect raw TCP socket, send queued request, verify it entered queue, then drop socket
            sock = _open_raw_post(port, "k0", "tcp_victim")
            _wait_rendered(render_calls, "tcp_victim")
            sock.close()

            # Because max_queue_size=1, survivor can only enter queue once tcp_victim has been evicted
            survivor_res = []
            accepted = False
            t_survivor = None
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline and not accepted:
                out: list[tuple] = []
                t_cand = threading.Thread(target=lambda o=out: o.append(_post_systemone(port, "k0", "survivor")))
                t_cand.start()
                # Wait briefly to see if it got rejected immediately with 429 or stayed waiting in queue
                t_cand.join(timeout=0.05)
                if t_cand.is_alive():
                    accepted = True
                    t_survivor = t_cand
                    survivor_res = out
                else:
                    assert out and out[0][0] == 429
            assert accepted and t_survivor is not None

            release_blocker.set()
            t_blocker.join(timeout=5.0)
            t_survivor.join(timeout=5.0)

            assert blocker_res[0][0] == 200
            assert survivor_res[0][0] == 200
            assert "tcp_victim" not in decider.eval_calls
            assert decider.eval_calls == ["blocker", "survivor"]
            assert uvicorn_errors == []

        captured = capsys.readouterr().out
        assert "499" in captured and "cancelled" in captured
        assert "No response returned" not in captured
    finally:
        logging.getLogger("uvicorn.error").removeHandler(err_handler)


def test_multi_key_round_robin_under_contention(monkeypatch: pytest.MonkeyPatch):
    render_calls: list[str] = []
    _install_fake_hindsight_decision(monkeypatch, render_calls)
    decider = _FakeDecider()
    blocker_started, release_blocker = decider.arm("blocker")

    app = srv.create_app(
        decider,
        "hs-decide-1-flash",
        None,
        ["k0", "k1"],
        0,
        use_color=False,
        max_queue_size=10,
        max_queue_per_key=10,
        queue_timeout_s=10.0,
    )

    with _running_uvicorn(app, "k0") as port:
        threads = []
        t_b = threading.Thread(target=lambda: _post_systemone(port, "k0", "blocker"))
        t_b.start()
        threads.append(t_b)
        assert blocker_started.wait(timeout=3.0)

        for key, tag in [("k0", "k0_1"), ("k0", "k0_2"), ("k1", "k1_1"), ("k1", "k1_2")]:
            t = threading.Thread(target=lambda k=key, tg=tag: _post_systemone(port, k, tg))
            t.start()
            threads.append(t)
            _wait_rendered(render_calls, tag)

        release_blocker.set()
        for t in threads:
            t.join(timeout=5.0)

        # After blocker, FairQueue alternates k0 and k1 rather than draining k0_1, k0_2 back-to-back
        assert decider.eval_calls[0] == "blocker"
        assert decider.eval_calls[1:] == ["k0_1", "k1_1", "k0_2", "k1_2"]


def test_batch_mode_push_front_and_zero_disables_limits(monkeypatch: pytest.MonkeyPatch):
    render_calls: list[str] = []
    _install_fake_hindsight_decision(monkeypatch, render_calls)
    decider = _FakeDecider()
    decider.max_batch_size = 2
    decider.max_batch_tokens = 200
    decider.ctx_size = 200
    blocker_started, release_blocker = decider.arm("blocker")

    # Zero queue limits & zero timeout disable bounds cleanly; batch_wait_ms > 0 exercises wait loop & shutdown
    app = srv.create_app(
        decider,
        "hs-decide-1-flash",
        None,
        ["k0", "k1"],
        batch_wait_ms=80,
        use_color=False,
        max_queue_size=0,
        max_queue_per_key=0,
        queue_timeout_s=0.0,
    )

    with _running_uvicorn(app, "k0") as port:
        results = {}
        threads = []

        t_b = threading.Thread(target=lambda: results.__setitem__("blocker", _post_systemone(port, "k0", "blocker", tokens=120)[:3:2]))
        t_b.start()
        threads.append(t_b)
        assert blocker_started.wait(timeout=3.0)

        # Both k0_big (120 tok) and k1_big (120 tok) exceed max_batch_tokens=200 together, forcing push_front on k1_big
        for key, tag in [("k0", "k0_big"), ("k1", "k1_big"), ("k0", "k0_next")]:
            t = threading.Thread(target=lambda k=key, tg=tag: results.__setitem__(tg, _post_systemone(port, k, tg, tokens=120)[:3:2]))
            t.start()
            threads.append(t)
            _wait_rendered(render_calls, tag)

        release_blocker.set()
        for t in threads:
            t.join(timeout=5.0)

        for tag in ("blocker", "k0_big", "k1_big", "k0_next"):
            assert results[tag][0] == 200
        # k1_big was pushed back to front of ring ahead of k0_next
        assert decider.eval_calls == ["blocker", "k0_big", "k1_big", "k0_next"]


def test_phase2_disconnect_eval_error_500_and_error_headers(monkeypatch: pytest.MonkeyPatch):
    render_calls: list[str] = []
    _install_fake_hindsight_decision(monkeypatch, render_calls)
    decider = _FakeDecider()
    gpu_started, release_gpu = decider.arm("mid_gpu")

    app = srv.create_app(
        decider,
        "hs-decide-1-flash",
        None,
        ["k0"],
        0,
        use_color=False,
        max_queue_size=4,
        max_queue_per_key=4,
        queue_timeout_s=5.0,
    )

    with _running_uvicorn(app, "k0") as port:
        # 1. 401 unauthorized carries no timing or batch headers
        st_401, hdr_401, _ = _post_systemone(port, "wrong", "unauth")
        assert st_401 == 401
        assert (
            "server-timing" not in hdr_401
            and "x-decide-queue-ms" not in hdr_401
            and "x-decide-batch-size" not in hdr_401
            and "x-decide-batch-id" not in hdr_401
        )

        # 2. Client disconnects AFTER GPU eval starts (Phase 2)
        sock = _open_raw_post(port, "k0", "mid_gpu")
        assert gpu_started.wait(timeout=3.0)
        sock.close()
        time.sleep(0.05)
        release_gpu.set()

        # Server still handles subsequent requests cleanly; eval_batch failure propagates cleanly
        def boom(_rendered):
            raise RuntimeError("cuda boom")

        monkeypatch.setattr(decider, "eval_batch", boom)
        st_500, hdr_500, _ = _post_systemone(port, "k0", "boom")
        assert st_500 == 500
        assert (
            "server-timing" not in hdr_500
            and "x-decide-queue-ms" not in hdr_500
            and "x-decide-batch-size" not in hdr_500
            and "x-decide-batch-id" not in hdr_500
        )


def test_main_env_queue_defaults(monkeypatch: pytest.MonkeyPatch):
    captured = {}

    class _DummyDecider:
        def __init__(self, *_a, **_kw):
            pass

        def engine_version(self):
            return "dummy"

    def fake_create_app(*args, **kwargs):
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setenv("DECIDE_MAX_QUEUE_SIZE", "48")
    monkeypatch.setenv("DECIDE_MAX_QUEUE_PER_KEY", "12")
    monkeypatch.setenv("DECIDE_QUEUE_TIMEOUT_S", "45.5")
    monkeypatch.setattr(srv, "LlamaCppDecider", _DummyDecider)
    monkeypatch.setattr(srv, "create_app", fake_create_app)

    fake_uvicorn = types.ModuleType("uvicorn")
    fake_uvicorn.run = lambda app, **kw: captured.setdefault("uvicorn", kw)
    monkeypatch.setitem(sys.modules, "uvicorn", fake_uvicorn)

    srv.main(["/model", "--gguf", "/model/m.gguf", "--api-key", "k0"])
    assert captured["kwargs"]["max_queue_size"] == 48
    assert captured["kwargs"]["max_queue_per_key"] == 12
    assert captured["kwargs"]["queue_timeout_s"] == 45.5


def test_batch_headers_multi_seq_and_serial_fallback(monkeypatch: pytest.MonkeyPatch, capsys):
    render_calls: list[str] = []
    _install_fake_hindsight_decision(monkeypatch, render_calls)

    # Part A: real multi-sequence batch shares batch_id and reports batch_size=2
    decider = _FakeDecider()
    decider.max_batch_size = 4
    blocker_started, release_blocker = decider.arm("blocker")
    app = srv.create_app(
        decider,
        "hs-decide-1-flash",
        None,
        ["k0"],
        batch_wait_ms=0,
        use_color=False,
        max_queue_size=10,
        max_queue_per_key=10,
        queue_timeout_s=10.0,
    )

    with _running_uvicorn(app, "k0") as port:
        results = {}
        threads = []
        t_b = threading.Thread(target=lambda: results.__setitem__("blocker", _post_systemone(port, "k0", "blocker", tokens=100)))
        t_b.start()
        threads.append(t_b)
        assert blocker_started.wait(timeout=3.0)

        for tag, toks in [("req_short", 100), ("req_long", 400)]:
            t = threading.Thread(
                target=lambda tg=tag, tk=toks: results.__setitem__(tg, _post_systemone(port, "k0", tg, tokens=tk))
            )
            t.start()
            threads.append(t)
            _wait_rendered(render_calls, tag)

        release_blocker.set()
        for t in threads:
            t.join(timeout=5.0)

        st_b, hdr_b, _ = results["blocker"]
        st_s, hdr_s, data_s = results["req_short"]
        st_l, hdr_l, data_l = results["req_long"]
        assert (st_b, st_s, st_l) == (200, 200, 200)
        assert hdr_b["x-decide-batch-size"] == "1"
        assert hdr_s["x-decide-batch-size"] == "2"
        assert hdr_l["x-decide-batch-size"] == "2"
        assert hdr_s["x-decide-batch-id"] == hdr_l["x-decide-batch-id"]
        assert hdr_s["x-decide-batch-id"] != hdr_b["x-decide-batch-id"]
        assert data_s["tag"] == "req_short" and data_l["tag"] == "req_long"

    # Drain Part A stdout before Part B fallback check
    capsys.readouterr()

    # Part B: real LlamaCppDecider.eval_batch serial fallback must report batch_size=1 & distinct batch_ids
    render_calls.clear()
    fb_decider = object.__new__(srv.LlamaCppDecider)
    fb_decider.folder = Path(".")
    fb_decider.settings = types.SimpleNamespace(max_request_tokens=16384)
    fb_decider.tokenizer = None
    fb_decider.ids = [1, 2]
    fb_decider.ctx_size = 16384
    fb_decider.max_batch_tokens = 16384
    fb_decider.max_batch_size = 4
    fb_decider.max_outputs = 256
    fb_decider.device_label = "cpu"
    fb_decider.last_batch_fell_back = False

    fb_started = threading.Event()
    fb_release = threading.Event()
    multi_attempts: list[list[str]] = []
    serial_evals: list[str] = []

    def fake_eval_once(rendered_list):
        if len(rendered_list) > 1:
            multi_attempts.append([r.tag for r in rendered_list])
            raise RuntimeError("simulated multi-seq eval failure")
        item = rendered_list[0]
        serial_evals.append(item.tag)
        if item.tag == "fb_blocker":
            fb_started.set()
            fb_release.wait(timeout=10.0)
        return [[[0.0, 1.0] for _ in item.questions]]

    fb_decider._eval_once = fake_eval_once

    fb_app = srv.create_app(
        fb_decider,
        "hs-decide-1-flash",
        None,
        ["k0"],
        batch_wait_ms=0,
        use_color=False,
        max_queue_size=10,
        max_queue_per_key=10,
        queue_timeout_s=10.0,
    )

    with _running_uvicorn(fb_app, "k0") as port:
        fb_results = {}
        fb_threads = []
        t_b = threading.Thread(
            target=lambda: fb_results.__setitem__("fb_blocker", _post_systemone(port, "k0", "fb_blocker", tokens=100))
        )
        t_b.start()
        fb_threads.append(t_b)
        assert fb_started.wait(timeout=3.0)

        for tag, toks in [("fb_short", 100), ("fb_long", 400)]:
            t = threading.Thread(
                target=lambda tg=tag, tk=toks: fb_results.__setitem__(tg, _post_systemone(port, "k0", tg, tokens=tk))
            )
            t.start()
            fb_threads.append(t)
            _wait_rendered(render_calls, tag)

        fb_release.set()
        for t in fb_threads:
            t.join(timeout=5.0)

        assert len(multi_attempts) == 1
        assert set(multi_attempts[0]) == {"fb_short", "fb_long"}
        assert serial_evals == ["fb_blocker", "fb_short", "fb_long"]

        st_fs, hdr_fs, _ = fb_results["fb_short"]
        st_fl, hdr_fl, _ = fb_results["fb_long"]
        assert st_fs == 200 and st_fl == 200
        assert hdr_fs["x-decide-batch-size"] == "1"
        assert hdr_fl["x-decide-batch-size"] == "1"
        assert hdr_fs["x-decide-batch-id"] != hdr_fl["x-decide-batch-id"]

    captured = capsys.readouterr().out
    assert "b=2" not in captured


def test_batch_worker_respects_ctx_size_when_smaller_than_max_batch_tokens(monkeypatch: pytest.MonkeyPatch):
    render_calls: list[str] = []
    _install_fake_hindsight_decision(monkeypatch, render_calls)
    decider = _FakeDecider()
    decider.max_batch_size = 2
    decider.max_batch_tokens = 1000
    decider.ctx_size = 200
    blocker_started, release_blocker = decider.arm("blocker")

    app = srv.create_app(
        decider,
        "hs-decide-1-flash",
        None,
        ["k0"],
        batch_wait_ms=0,
        use_color=False,
        max_queue_size=10,
        max_queue_per_key=10,
        queue_timeout_s=10.0,
    )

    with _running_uvicorn(app, "k0") as port:
        results = {}
        threads = []
        t_b = threading.Thread(
            target=lambda: results.__setitem__("blocker", _post_systemone(port, "k0", "blocker", tokens=120))
        )
        t_b.start()
        threads.append(t_b)
        assert blocker_started.wait(timeout=3.0)

        for tag in ("a", "b"):
            t = threading.Thread(
                target=lambda tg=tag: results.__setitem__(tg, _post_systemone(port, "k0", tg, tokens=120))
            )
            t.start()
            threads.append(t)
            _wait_rendered(render_calls, tag)

        release_blocker.set()
        for t in threads:
            t.join(timeout=5.0)

        st_a, hdr_a, _ = results["a"]
        st_b, hdr_b, _ = results["b"]
        assert st_a == 200 and st_b == 200
        assert hdr_a["x-decide-batch-size"] == "1"
        assert hdr_b["x-decide-batch-size"] == "1"
        assert hdr_a["x-decide-batch-id"] != hdr_b["x-decide-batch-id"]
