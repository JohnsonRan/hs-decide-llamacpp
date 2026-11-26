"""``hindsight-decision-serve-llamacpp``: ``POST /v1/systemone`` backed by llama.cpp (Q8_0)
with multi-key auth and per-key round-robin fair queueing.
"""

import argparse
import asyncio
import ctypes
import json
import logging
import os
import secrets
import subprocess
import time
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("hindsight_decision_llamacpp")

SERVER = "hindsight-decision-serve-llamacpp"
MODEL_NAME = "hs-decide-1-flash"
DEFAULT_SHIM_LIB = "/opt/llamacpp/lib/libhs_decide_llama.so"


def parse_api_keys(primary: str | None, extra_csv: str | None) -> list[str]:
    """Combine primary and comma-separated extra API keys, deduplicated in order."""
    keys: list[str] = []
    seen: set[str] = set()
    candidates: list[str] = []
    if primary:
        candidates.append(primary)
    if extra_csv:
        candidates.extend(extra_csv.split(","))
    for raw in candidates:
        k = raw.strip()
        if k and k not in seen:
            seen.add(k)
            keys.append(k)
    return keys


def match_tenant(auth_header: str, keys: list[str]) -> str | None:
    """Return an opaque tenant id (``k0``, ``k1``, ...) on Bearer match, or None."""
    if not keys:
        return "anon"
    matched: str | None = None
    for idx, k in enumerate(keys):
        if secrets.compare_digest(auth_header, f"Bearer {k}"):
            matched = f"k{idx}"
    return matched


@dataclass
class PendingItem:
    request: Any
    rendered: Any
    future: asyncio.Future[dict[str, Any]]


class FairQueue:
    """Per-tenant FIFO queues drained in round-robin order so one key cannot starve others."""

    def __init__(self) -> None:
        self._queues: dict[str, deque[PendingItem]] = {}
        self._ring: deque[str] = deque()
        self._in_ring: set[str] = set()

    def total_pending(self) -> int:
        return sum(len(q) for q in self._queues.values())

    def push(self, tenant_id: str, item: PendingItem) -> None:
        q = self._queues.get(tenant_id)
        if q is None:
            q = deque()
            self._queues[tenant_id] = q
        q.append(item)
        if tenant_id not in self._in_ring:
            self._in_ring.add(tenant_id)
            self._ring.append(tenant_id)

    def push_front(self, tenant_id: str, item: PendingItem) -> None:
        q = self._queues.get(tenant_id)
        if q is None:
            q = deque()
            self._queues[tenant_id] = q
        q.appendleft(item)
        if tenant_id in self._in_ring:
            self._ring.remove(tenant_id)
        self._in_ring.add(tenant_id)
        self._ring.appendleft(tenant_id)

    def pop_next(self) -> tuple[str, PendingItem] | None:
        while self._ring:
            tid = self._ring.popleft()
            q = self._queues.get(tid)
            while q and q[0].future.cancelled():
                q.popleft()
            if not q:
                self._queues.pop(tid, None)
                self._in_ring.discard(tid)
                continue
            item = q.popleft()
            while q and q[0].future.cancelled():
                q.popleft()
            if q:
                self._ring.append(tid)
            else:
                self._queues.pop(tid, None)
                self._in_ring.discard(tid)
            return tid, item
        return None


def load_shim(lib_path: str) -> ctypes.CDLL:
    lib = ctypes.CDLL(lib_path)
    lib.hs_engine_create.argtypes = [
        ctypes.c_char_p,  # model_path
        ctypes.c_int32,   # n_ctx
        ctypes.c_int32,   # n_batch
        ctypes.c_int32,   # n_ubatch
        ctypes.c_int32,   # n_seq_max
        ctypes.c_int32,   # n_outputs_max
        ctypes.c_int32,   # n_gpu_layers
        ctypes.c_int32,   # n_threads
        ctypes.c_int32,   # flash_attn_mode
    ]
    lib.hs_engine_create.restype = ctypes.c_void_p

    lib.hs_engine_vocab_size.argtypes = [ctypes.c_void_p]
    lib.hs_engine_vocab_size.restype = ctypes.c_int32

    lib.hs_engine_eval_batch.argtypes = [
        ctypes.c_void_p,                  # eng
        ctypes.c_int32,                   # n_seqs
        ctypes.POINTER(ctypes.c_int32),   # seq_lens
        ctypes.POINTER(ctypes.c_int32),   # token_ids
        ctypes.POINTER(ctypes.c_int32),   # seq_slot_counts
        ctypes.POINTER(ctypes.c_int32),   # slot_positions
        ctypes.POINTER(ctypes.c_int32),   # classifier_ids
        ctypes.c_int32,                   # n_cls
        ctypes.POINTER(ctypes.c_float),   # out_logits
    ]
    lib.hs_engine_eval_batch.restype = ctypes.c_int32

    lib.hs_engine_destroy.argtypes = [ctypes.c_void_p]
    lib.hs_engine_destroy.restype = None
    return lib


def limited(settings: Any, max_request_tokens: int | None) -> Any:
    if max_request_tokens is None:
        return settings
    if not 0 < max_request_tokens <= settings.max_request_tokens:
        raise ValueError(
            f"max_request_tokens must be 1 to {settings.max_request_tokens}, not {max_request_tokens}"
        )
    return settings.model_copy(update={"max_request_tokens": max_request_tokens})


def detect_device_label(gpu_layers: int) -> str:
    if gpu_layers == 0:
        return "cpu"
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
            timeout=3,
        ).decode("utf-8", errors="replace").strip()
        if out:
            return out.splitlines()[0].strip()
    except Exception:
        pass
    return "cuda"


class LlamaCppDecider:
    def __init__(
        self,
        folder: Path,
        gguf_path: Path,
        *,
        shim_lib: str = DEFAULT_SHIM_LIB,
        max_request_tokens: int | None = None,
        ctx_size: int = 8192,
        ubatch_size: int = 1024,
        max_batch_size: int = 1,
        max_batch_tokens: int = 8192,
        max_outputs: int = 256,
        gpu_layers: int = -1,
        threads: int = 4,
        flash_attn: str = "auto",
    ) -> None:
        from hindsight_decision.prompt import load_settings
        from hindsight_decision.typed import classifier_ids
        from transformers import AutoTokenizer

        self.folder = folder
        self.gguf_path = gguf_path
        self.settings = limited(load_settings(folder), max_request_tokens)
        self.tokenizer = AutoTokenizer.from_pretrained(folder)
        self.ids = classifier_ids(self.tokenizer)
        self.n_cls = len(self.ids)
        self._cls_arr = (ctypes.c_int32 * self.n_cls)(*self.ids)
        self.ctx_size = ctx_size
        self.ubatch_size = ubatch_size
        self.max_batch_size = max(1, max_batch_size)
        self.max_batch_tokens = max_batch_tokens
        self.max_outputs = max_outputs
        self.gpu_layers = gpu_layers

        fa_mode = {"auto": -1, "off": 0, "on": 1}[flash_attn]
        self._lib = load_shim(shim_lib)
        self._eng = self._lib.hs_engine_create(
            str(gguf_path).encode("utf-8"),
            ctx_size,
            max_batch_tokens,
            ubatch_size,
            self.max_batch_size,
            max_outputs,
            gpu_layers,
            threads,
            fa_mode,
        )
        if not self._eng:
            raise RuntimeError(f"Failed to initialize llama.cpp engine from {gguf_path}")
        vocab_size = self._lib.hs_engine_vocab_size(self._eng)
        if max(self.ids) >= vocab_size:
            raise RuntimeError(f"Classifier token id {max(self.ids)} >= gguf vocab {vocab_size}")
        self.device_label = detect_device_label(gpu_layers)

    def close(self) -> None:
        if getattr(self, "_eng", None):
            self._lib.hs_engine_destroy(self._eng)
            self._eng = None

    def engine_version(self) -> str:
        llama_tag = os.environ.get("LLAMA_CPP_TAG", "v1.0.0")
        quant_desc = os.environ.get("QUANT_DESC", "Q8_0")
        return (
            f"{SERVER}; llama.cpp {llama_tag}; {self.device_label}; "
            f"{quant_desc}; batch<={self.max_batch_size}; ctx={self.ctx_size}; fair_rr"
        )

    def _eval_once(self, rendered_list: list[Any]) -> list[list[list[float]]]:
        n_seqs = len(rendered_list)
        seq_lens = [r.tokens() for r in rendered_list]
        flat_tokens = [tok for r in rendered_list for tok in r.ids]
        slot_counts = [len(r.questions) for r in rendered_list]
        flat_slots = [q.slot for r in rendered_list for q in r.questions]
        total_slots = sum(slot_counts)

        c_seq_lens = (ctypes.c_int32 * n_seqs)(*seq_lens)
        c_tokens = (ctypes.c_int32 * len(flat_tokens))(*flat_tokens)
        c_slot_counts = (ctypes.c_int32 * n_seqs)(*slot_counts)
        c_slots = (ctypes.c_int32 * total_slots)(*flat_slots)
        c_out = (ctypes.c_float * (total_slots * self.n_cls))()

        rc = self._lib.hs_engine_eval_batch(
            self._eng,
            n_seqs,
            c_seq_lens,
            c_tokens,
            c_slot_counts,
            c_slots,
            self._cls_arr,
            self.n_cls,
            c_out,
        )
        if rc != 0:
            raise RuntimeError(f"hs_engine_eval_batch failed with rc={rc} (n_seqs={n_seqs}, tokens={sum(seq_lens)})")

        flat_out = list(c_out)
        per_req: list[list[list[float]]] = []
        pos = 0
        for count in slot_counts:
            req_rows: list[list[float]] = []
            for _ in range(count):
                req_rows.append(flat_out[pos : pos + self.n_cls])
                pos += self.n_cls
            per_req.append(req_rows)
        return per_req

    def eval_batch(self, rendered_list: list[Any]) -> list[list[list[float]]]:
        try:
            return self._eval_once(rendered_list)
        except Exception as err:
            if len(rendered_list) <= 1:
                raise
            logger.warning("Multi-sequence batch failed (%s); falling back to serial", err)
            return [self._eval_once([item])[0] for item in rendered_list]


def create_app(
    decider: LlamaCppDecider,
    model: str,
    revision: str | None,
    api_keys: list[str],
    batch_wait_ms: int,
) -> Any:
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    from hindsight_decision.endpoint import answer_body
    from hindsight_decision.identity import identity, models_with_identity
    from hindsight_decision.typed import response
    from hindsight_decision.typed_render import render_request

    served = identity(model, decider.folder, revision)
    engine_str = decider.engine_version()

    fair_q = FairQueue()
    cond = asyncio.Condition()
    stop_flag = False

    async def batch_worker() -> None:
        while True:
            async with cond:
                while fair_q.total_pending() == 0 and not stop_flag:
                    await cond.wait()
                if stop_flag and fair_q.total_pending() == 0:
                    return

                if decider.max_batch_size > 1 and batch_wait_ms > 0 and fair_q.total_pending() < decider.max_batch_size:
                    deadline = time.monotonic() + (batch_wait_ms / 1000.0)
                    while fair_q.total_pending() < decider.max_batch_size:
                        rem = deadline - time.monotonic()
                        if rem <= 0:
                            break
                        try:
                            await asyncio.wait_for(cond.wait(), timeout=rem)
                        except asyncio.TimeoutError:
                            break

                batch: list[PendingItem] = []
                batch_tokens = 0
                batch_slots = 0
                while len(batch) < decider.max_batch_size:
                    popped = fair_q.pop_next()
                    if popped is None:
                        break
                    tid, nxt = popped
                    t = nxt.rendered.tokens()
                    s = len(nxt.rendered.questions)
                    if batch and (
                        batch_tokens + t > decider.max_batch_tokens
                        or batch_slots + s > decider.max_outputs
                    ):
                        fair_q.push_front(tid, nxt)
                        break
                    batch.append(nxt)
                    batch_tokens += t
                    batch_slots += s

            if not batch:
                continue

            try:
                rows_list = await asyncio.to_thread(
                    decider.eval_batch, [item.rendered for item in batch]
                )
                for item, rows in zip(batch, rows_list, strict=True):
                    if not item.future.done():
                        item.future.set_result(
                            response(item.request, item.rendered, rows, decider.ids)
                        )
            except Exception as exc:
                for item in batch:
                    if not item.future.done():
                        item.future.set_exception(exc)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        nonlocal stop_flag
        task = asyncio.create_task(batch_worker())
        try:
            yield
        finally:
            async with cond:
                stop_flag = True
                cond.notify_all()
            await task
            decider.close()

    server = FastAPI(title=SERVER, lifespan=lifespan)

    @server.middleware("http")
    async def check_key_and_report_identity(raw_request: Request, call_next: Any) -> Any:
        given = raw_request.headers.get("authorization", "")
        tenant = match_tenant(given, api_keys)
        if tenant is None:
            return JSONResponse({"error": "Unauthorized"}, status_code=401)
        raw_request.state.tenant_id = tenant
        resp = await call_next(raw_request)
        resp.headers.update(served.headers())
        return resp

    async def respond_for(request: Any, tenant_id: str) -> dict[str, Any]:
        rendered = render_request(decider.tokenizer, request, decider.settings)
        if rendered.tokens() > decider.max_batch_tokens:
            raise RuntimeError(
                f"rendered tokens {rendered.tokens()} > max_batch_tokens {decider.max_batch_tokens}"
            )
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[dict[str, Any]] = loop.create_future()
        async with cond:
            fair_q.push(tenant_id, PendingItem(request=request, rendered=rendered, future=fut))
            cond.notify()
        return await asyncio.shield(fut)

    @server.post("/v1/systemone")
    async def systemone(raw_request: Request) -> Any:
        tenant_id = getattr(raw_request.state, "tenant_id", "anon")
        return await answer_body(raw_request, lambda req: respond_for(req, tenant_id))

    @server.get("/v1/models")
    async def models() -> Any:
        listing = {
            "object": "list",
            "data": [
                {
                    "id": MODEL_NAME,
                    "object": "model",
                    "owned_by": SERVER,
                    "root": model,
                    "max_model_len": decider.settings.max_request_tokens,
                }
            ],
        }
        return JSONResponse(json.loads(models_with_identity(json.dumps(listing).encode(), served)))

    @server.get("/version")
    async def engine_report() -> dict[str, str]:
        return {"version": engine_str}

    @server.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return server


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog=SERVER)
    parser.add_argument("model_dir")
    parser.add_argument("--gguf", required=True)
    parser.add_argument("--shim-lib", default=DEFAULT_SHIM_LIB)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--extra-api-keys", default=os.environ.get("DECIDE_EXTRA_API_KEYS", ""))
    parser.add_argument("--max-request-tokens", type=int, default=None)
    parser.add_argument("--ctx-size", type=int, default=8192)
    parser.add_argument("--ubatch-size", type=int, default=1024)
    parser.add_argument("--max-batch-size", type=int, default=1)
    parser.add_argument("--max-batch-tokens", type=int, default=8192)
    parser.add_argument("--max-outputs", type=int, default=256)
    parser.add_argument("--batch-wait-ms", type=int, default=0)
    parser.add_argument("--gpu-layers", type=int, default=-1)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--flash-attn", choices=["auto", "on", "off"], default="auto")
    args = parser.parse_args(argv)

    import uvicorn

    api_keys = parse_api_keys(args.api_key, args.extra_api_keys)
    decider = LlamaCppDecider(
        Path(args.model_dir),
        Path(args.gguf),
        shim_lib=args.shim_lib,
        max_request_tokens=args.max_request_tokens,
        ctx_size=args.ctx_size,
        ubatch_size=args.ubatch_size,
        max_batch_size=args.max_batch_size,
        max_batch_tokens=args.max_batch_tokens,
        max_outputs=args.max_outputs,
        gpu_layers=args.gpu_layers,
        threads=args.threads,
        flash_attn=args.flash_attn,
    )
    print(f"Serving {args.gguf} ({decider.engine_version()}, keys={len(api_keys)})", flush=True)
    uvicorn.run(
        create_app(decider, args.model_dir, args.revision, api_keys, args.batch_wait_ms),
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
