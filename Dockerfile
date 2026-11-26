# llama.cpp (Q8_0 + F32 output head) image for hs-decide-1-flash on NVIDIA GPUs (including Pascal sm_61).
FROM nvidia/cuda:12.6.3-devel-ubuntu24.04 AS builder

ARG HS_DECIDE_REPO=hindsight-ai/hs-decide-1-flash
ARG HS_DECIDE_REVISION=e2c789026dfd1f7608cafb2ef6a9bb31397af802
ARG LLAMA_CPP_TAG=v1.0.0
ARG CUDA_ARCH=61
ARG BUILD_JOBS=3
ARG QUANT_TYPE=Q8_0
ARG OUTPUT_TENSOR_TYPE=f32

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_TELEMETRY=1

# Remove NVIDIA's apt list so temporary CDN mirror sync mismatches cannot break Ubuntu package installs.
RUN rm -f /etc/apt/sources.list.d/cuda* \
 && apt-get update && apt-get install -y --no-install-recommends \
      git cmake ninja-build build-essential ca-certificates \
      python3 python3-pip python3-venv \
 && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch "${LLAMA_CPP_TAG}" https://github.com/ggml-org/llama.cpp.git /src/llama.cpp

# Build libllama, libggml (with CUDA MMQ forced on for DP4A/INT8), and llama-quantize.
RUN ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 \
 && LD_LIBRARY_PATH="/usr/local/cuda/lib64/stubs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    cmake -S /src/llama.cpp -B /src/llama.cpp/build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_SHARED_LIBS=ON \
      -DGGML_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" \
      -DGGML_CUDA_FORCE_MMQ=ON \
      -DGGML_CUDA_NCCL=OFF \
      -DGGML_NATIVE=OFF \
      -DLLAMA_BUILD_TESTS=OFF \
      -DLLAMA_BUILD_EXAMPLES=OFF \
      -DLLAMA_BUILD_SERVER=OFF \
      -DLLAMA_BUILD_APP=OFF \
      -DLLAMA_CURL=OFF \
      -DCMAKE_EXE_LINKER_FLAGS="-Wl,-rpath-link,/usr/local/cuda/lib64/stubs" \
 && LD_LIBRARY_PATH="/usr/local/cuda/lib64/stubs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    cmake --build /src/llama.cpp/build --target llama llama-quantize -j "${BUILD_JOBS}"

COPY src/hs_llama_shim.cpp /src/hs_llama_shim.cpp

RUN mkdir -p /opt/llamacpp/lib /opt/llamacpp/bin \
 && find /src/llama.cpp/build -name 'lib*.so*' ! -path '*/CMakeFiles/*' -exec cp -an {} /opt/llamacpp/lib/ \; \
 && cp -a /src/llama.cpp/build/bin/llama-quantize /opt/llamacpp/bin/ \
 && LD_LIBRARY_PATH="/opt/llamacpp/lib:/usr/local/cuda/lib64/stubs" \
    g++ -O3 -shared -fPIC -std=c++17 \
      -I/src/llama.cpp/include -I/src/llama.cpp/ggml/include \
      /src/hs_llama_shim.cpp \
      -L/opt/llamacpp/lib -Wl,-rpath,'$ORIGIN' \
      -Wl,-rpath-link,/opt/llamacpp/lib -Wl,-rpath-link,/usr/local/cuda/lib64/stubs \
      -lllama -lggml \
      -o /opt/llamacpp/lib/libhs_decide_llama.so

# Separate converter venv so PyTorch stays out of the runtime image.
RUN python3 -m venv /opt/conv-venv \
 && /opt/conv-venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch \
 && /opt/conv-venv/bin/pip install "huggingface-hub>=1.0" "transformers>=5.16.1" safetensors numpy sentencepiece protobuf \
 && /opt/conv-venv/bin/pip install /src/llama.cpp/gguf-py

# Download pinned weights, convert to bf16 GGUF, quantize to Q8_0 (preserving F32 tied output/embd
# and SSM tensors), then delete intermediate files in the same layer to keep disk usage small.
RUN /opt/conv-venv/bin/hf download "$HS_DECIDE_REPO" --revision "$HS_DECIDE_REVISION" --local-dir /opt/hs-decide \
      --include "*.json" --include "*.safetensors" --include "*.jinja" --include "serving/*" \
 && test -f /opt/hs-decide/decision.json && test -f /opt/hs-decide/config.json \
 && rm -rf /opt/hs-decide/.cache \
 && /opt/conv-venv/bin/python /src/llama.cpp/convert_hf_to_gguf.py /opt/hs-decide \
      --outfile /tmp/model-bf16.gguf --outtype bf16 --no-mtp \
 && LD_LIBRARY_PATH="/opt/llamacpp/lib:/usr/local/cuda/lib64/stubs" \
      /opt/llamacpp/bin/llama-quantize \
      --output-tensor-type "${OUTPUT_TENSOR_TYPE}" \
      --token-embedding-type "${OUTPUT_TENSOR_TYPE}" \
      /tmp/model-bf16.gguf /opt/hs-decide/model.gguf "${QUANT_TYPE}" \
 && rm -rf /tmp/model-bf16.gguf /opt/hs-decide/*.safetensors

FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04

ARG HS_DECIDE_REVISION=e2c789026dfd1f7608cafb2ef6a9bb31397af802
ARG LLAMA_CPP_TAG=v1.0.0
ARG QUANT_TYPE=Q8_0
ARG OUTPUT_TENSOR_TYPE=f32

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    PATH="/opt/venv/bin:$PATH" \
    HS_DECIDE_REVISION="${HS_DECIDE_REVISION}" \
    LLAMA_CPP_TAG="${LLAMA_CPP_TAG}" \
    QUANT_DESC="${QUANT_TYPE}_out_${OUTPUT_TENSOR_TYPE}"

RUN rm -f /etc/apt/sources.list.d/cuda* \
 && apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-venv libgomp1 ca-certificates \
 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/llamacpp/lib /opt/llamacpp/lib
RUN echo /opt/llamacpp/lib > /etc/ld.so.conf.d/llamacpp.conf && ldconfig

COPY --from=builder /opt/hs-decide /opt/hs-decide

# Install hindsight-decision without [server] extra so torch is not pulled into runtime.
RUN python3 -m venv /opt/venv \
 && /opt/venv/bin/pip install /opt/hs-decide/serving "fastapi>=0.115" "uvicorn>=0.30"

COPY src/server.py /opt/hs-decide-server/server.py

RUN useradd --system --uid 10001 decide
USER decide

ENV DECIDE_MAX_REQUEST_TOKENS=8192 \
    DECIDE_CTX_SIZE=8192 \
    DECIDE_UBATCH_SIZE=1024 \
    DECIDE_MAX_BATCH_SIZE=1 \
    DECIDE_MAX_BATCH_TOKENS=8192 \
    DECIDE_MAX_OUTPUTS=256 \
    DECIDE_BATCH_WAIT_MS=0 \
    DECIDE_GPU_LAYERS=-1 \
    DECIDE_THREADS=4 \
    DECIDE_FLASH_ATTN=auto \
    DECIDE_EXTRA_API_KEYS="" \
    DECIDE_LOG_SLOW_MS=1000 \
    DECIDE_LOG_COLOR=always

EXPOSE 8000

CMD ["sh", "-c", "exec python /opt/hs-decide-server/server.py /opt/hs-decide --gguf /opt/hs-decide/model.gguf --revision \"$HS_DECIDE_REVISION\" --host 0.0.0.0 --port 8000 --api-key \"$DECIDE_API_KEY\" --extra-api-keys \"$DECIDE_EXTRA_API_KEYS\" --max-request-tokens \"$DECIDE_MAX_REQUEST_TOKENS\" --ctx-size \"$DECIDE_CTX_SIZE\" --ubatch-size \"$DECIDE_UBATCH_SIZE\" --max-batch-size \"$DECIDE_MAX_BATCH_SIZE\" --max-batch-tokens \"$DECIDE_MAX_BATCH_TOKENS\" --max-outputs \"$DECIDE_MAX_OUTPUTS\" --batch-wait-ms \"$DECIDE_BATCH_WAIT_MS\" --gpu-layers \"$DECIDE_GPU_LAYERS\" --threads \"$DECIDE_THREADS\" --flash-attn \"$DECIDE_FLASH_ATTN\""]
