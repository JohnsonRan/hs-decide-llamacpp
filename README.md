# hs-decide-llamacpp

Serve [`hindsight-ai/hs-decide-1-flash`](https://huggingface.co/hindsight-ai/hs-decide-1-flash) (`POST /v1/systemone`) via `llama.cpp` (`Q8_0` with `F32` output/embedding head) on NVIDIA GPUs — including **Pascal (`sm_61`) cards such as Tesla P4 / P40** that cannot run vLLM (`sm_70+`) or Triton-based `flash-linear-attention`.

## Why this exists

Upstream `hindsight-decision-serve-transformers` on Pascal (`sm_61`) must run in `float32` (Pascal has no `bfloat16` and `float16` runs at 1/64 rate) and falls back to plain PyTorch loops for Qwen3.5's 18 Gated DeltaNet linear-attention layers, using ~5.5 GiB VRAM for a 0.8B model.

This project replaces only the forward pass with a thin `ctypes` shim over `llama.cpp` (`v1.0.0`) while **reusing the upstream `hindsight_decision` package verbatim** for request validation, prompt rendering, slot extraction, and calibrated softmax scoring:

- **Unlocks Pascal INT8 DP4A (`__dp4a` / MMQ)** + native CUDA `GGML_OP_GATED_DELTA_NET` kernels.
- **Preserves accuracy**: `Q8_0` weights with `F32` `token_embd`/`lm_head` and `F32` SSM tensors (`--no-mtp` conversion).
- **Cuts VRAM by 63%**: `~2.0 GiB` peak VRAM (`ctx=8192`) vs `5.5 GiB` in Transformers FP32.
- **Multi-key fair queueing**: supports multiple Bearer API keys (`DECIDE_API_KEY` + `DECIDE_EXTRA_API_KEYS`) with per-key round-robin scheduling so burst traffic from one client cannot starve another.

## Measured Benchmarks (NVIDIA Tesla P4, 8 GB)

### Latency & Throughput (`c=1`)

| Input length | Transformers FP32 | `llama.cpp Q8_0` (`batch=1, ctx=8192`) | Speedup |
| ---: | --- | --- | ---: |
| 471 tok | `0.25s` · `1859 tok/s` · `3.90 req/s` | **`0.15s`** · **`3227 tok/s`** · **`6.75 req/s`** | **1.73×** |
| 1264 tok | `0.58s` · `2174 tok/s` · `1.72 req/s` | **`0.36s`** · **`3519 tok/s`** · **`2.79 req/s`** | **1.62×** |
| 3339 tok | `1.67s` · `2008 tok/s` · `0.60 req/s` | **`0.99s`** · **`3364 tok/s`** · **`1.01 req/s`** | **1.68×** |
| 7516 tok | `4.30s` · `1748 tok/s` · `0.23 req/s` | **`2.58s`** · **`2913 tok/s`** · **`0.39 req/s`** | **1.67×** |
| **VRAM** | `5507 MiB / 7680 MiB` | **`1979 MiB` idle / `2007 MiB` peak** | **-63.6%** |

### Accuracy vs Transformers FP32 (`Q8_0_out_f32`, 14 EN/ZH cases, 202–6666 tokens)

- **Choice Top-1 Agreement**: `14 / 14 (100.00%)` (0 flips across `choice` and `noul >= 0.5`)
- **All Option Probabilities (`n=241`)**: `MAE = 0.00033`, `p95 = 0.00154`, `max = 0.00521`
- **`noul` (`n=22`)**: `MAE = 0.00095`, `max = 0.00356`
- **`score` expected (`n=13`)**: `MAE = 0.00141`, `max = 0.00687`

### Single-Instance Fair Queue vs Dual-Instance vs `llama_batch` Batching

On Tesla P4, a single `llama.cpp Q8_0` request already saturates GPU compute (`95–100%` utilization):
- **`llama_batch` multi-sequence batching (`batch<=2` / `batch<=4`) is slower for medium/long prompts** (`6.43s` vs `5.14s` at `c=2`, 7k tokens) because hybrid attention splits `ubatch` and adds scheduling overhead. Keep `DECIDE_MAX_BATCH_SIZE=1`.
- **Single instance (`compose.yaml`, default)**: Highest peak throughput (`2007 MiB` VRAM) with built-in **per-API-key round-robin fair queue** so multiple users alternate fairly even under burst load.
- **Dual instance (`compose.dual.yaml`, optional)**: Runs two independent `batch=1` containers (`~4.0 GiB` VRAM total) behind Caddy `least_conn`. Peak throughput is ~8% lower due to GPU time-slicing, **but short requests (~471 tok) complete in `0.33s` even while a `7516`-tok request is running on the other worker**.

## Quick Start

Requires Docker, `nvidia-container-toolkit`, and NVIDIA driver compatible with CUDA 12.6 (e.g. R580 branch for Tesla P4).

```sh
cp .env.example .env
# Edit .env and set DECIDE_API_KEY (e.g. `openssl rand -hex 32`)
# Optionally set DECIDE_EXTRA_API_KEYS for additional users, and CUDA_ARCH for your GPU:
#   61 = Tesla P4 / P40 / GTX 10xx (default)
#   75 = Tesla T4 / RTX 20xx
#   86 = RTX 30xx / A10
#   89 = RTX 40xx / L4

docker compose config --quiet
docker compose up -d --build --wait --wait-timeout 180
```

Verify health & engine version:

```sh
curl -fsS -H "Authorization: Bearer $DECIDE_API_KEY" http://127.0.0.1:8000/health
curl -fsS -H "Authorization: Bearer $DECIDE_API_KEY" http://127.0.0.1:8000/version
```

### Dual-Instance Mode (Optional)

If you want short requests isolated from concurrent 7k-token requests on the same GPU (>= 6 GB VRAM):

```sh
docker compose -f compose.dual.yaml up -d --build --wait --wait-timeout 180
```

## Multi-User Sharing & Hindsight Configuration

Give yourself `DECIDE_API_KEY` and each friend a separate key in `DECIDE_EXTRA_API_KEYS` (comma-separated):

```sh
DECIDE_API_KEY=<key-user-1>
DECIDE_EXTRA_API_KEYS=<key-user-2>,<key-user-3>
```

Each Bearer token gets its own FIFO queue drained round-robin. Keep `DECIDE_BIND_ADDRESS=127.0.0.1` unless exposing over a private mesh (WireGuard / Tailscale) or behind a TLS reverse proxy.

In each Hindsight host's `.env`:

```sh
HINDSIGHT_API_DECISION_MODEL_PROVIDER=hindsight
HINDSIGHT_API_DECISION_MODEL_BASE_URL=http://<decide-host>:8000
HINDSIGHT_API_DECISION_MODEL_API_KEY=<that-users-key>
HINDSIGHT_API_DECISION_MODEL_MAX_CONCURRENT=1
HINDSIGHT_API_DECISION_MODEL_MAX_REQUEST_TOKENS=7000
HINDSIGHT_API_DECISION_MODEL_TIMEOUT=180
```

The server defaults to `DECIDE_CTX_SIZE=16384` / `DECIDE_MAX_REQUEST_TOKENS=16384` (`~2.4 GiB` VRAM) while clients stay at `7000`: Hindsight's client-side token estimate often undercounts CJK/code prompts relative to the `Qwen3.5` tokenizer, so the `16384` server headroom absorbs `8k–12k` rendered bursts without triggering `400 max_tokens_exceeded` (`retrying at 3500 tokens`) split retries.

## Benchmarking & Accuracy Tools

Standard-library-only tools are included in `tools/`:

```sh
# Latency & throughput sweep
DECIDE_API_KEY=... python3 tools/bench.py --url http://127.0.0.1:8000 -n 5 -c 1

# Accuracy comparison against a reference FP32 server
DECIDE_API_KEY=... python3 tools/diff_accuracy.py --record ref.json --url http://<fp32-host>:8000
DECIDE_API_KEY=... python3 tools/diff_accuracy.py --compare ref.json --url http://127.0.0.1:8000
```

## License

Apache-2.0. Based on [hindsight-ai/hs-decide-1-flash](https://huggingface.co/hindsight-ai/hs-decide-1-flash) and [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp).
