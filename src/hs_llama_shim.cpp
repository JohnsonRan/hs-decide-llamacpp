#include "llama.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

struct HsLlamaEngine {
    llama_model * model;
    llama_context * ctx;
    int32_t n_vocab;
    int32_t n_ctx;
    int32_t n_batch;
    int32_t n_seq_max;
    int32_t n_outputs_max;
    std::mutex mu;
};

namespace {
std::once_flag g_backend_init_once;
}

extern "C" {

HsLlamaEngine * hs_engine_create(
    const char * model_path,
    int32_t n_ctx,
    int32_t n_batch,
    int32_t n_ubatch,
    int32_t n_seq_max,
    int32_t n_outputs_max,
    int32_t n_gpu_layers,
    int32_t n_threads,
    int32_t flash_attn_mode
) {
    if (!model_path || n_ctx <= 0 || n_batch <= 0 || n_ubatch <= 0 || n_seq_max <= 0 || n_outputs_max <= 0) {
        return nullptr;
    }

    std::call_once(g_backend_init_once, []() {
        llama_backend_init();
    });

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = n_gpu_layers;
    mparams.load_mtp = false;

    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        return nullptr;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = static_cast<uint32_t>(n_ctx);
    cparams.n_batch = static_cast<uint32_t>(n_batch);
    cparams.n_ubatch = static_cast<uint32_t>(n_ubatch);
    cparams.n_seq_max = static_cast<uint32_t>(n_seq_max);
    cparams.n_rs_seq = 0;
    cparams.n_outputs_max = static_cast<uint32_t>(n_outputs_max);
    cparams.n_threads = n_threads > 0 ? n_threads : 4;
    cparams.n_threads_batch = n_threads > 0 ? n_threads : 4;
    cparams.embeddings = false;
    cparams.kv_unified = true;
    cparams.offload_kqv = (n_gpu_layers != 0);
    cparams.op_offload = (n_gpu_layers != 0);
    if (flash_attn_mode < 0) {
        cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
    } else if (flash_attn_mode == 0) {
        cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    } else {
        cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    }

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        llama_model_free(model);
        return nullptr;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    int32_t n_vocab = llama_vocab_n_tokens(vocab);

    auto * eng = new HsLlamaEngine();
    eng->model = model;
    eng->ctx = ctx;
    eng->n_vocab = n_vocab;
    eng->n_ctx = n_ctx;
    eng->n_batch = n_batch;
    eng->n_seq_max = n_seq_max;
    eng->n_outputs_max = n_outputs_max;
    return eng;
}

int32_t hs_engine_vocab_size(const HsLlamaEngine * eng) {
    return eng ? eng->n_vocab : -1;
}

int32_t hs_engine_eval_batch(
    HsLlamaEngine * eng,
    int32_t n_seqs,
    const int32_t * seq_lens,
    const int32_t * token_ids,
    const int32_t * seq_slot_counts,
    const int32_t * slot_positions,
    const int32_t * classifier_ids,
    int32_t n_cls,
    float * out_logits
) {
    if (!eng || !seq_lens || !token_ids || !seq_slot_counts || !slot_positions || !classifier_ids || !out_logits) {
        return -10;
    }
    if (n_seqs < 1 || n_seqs > eng->n_seq_max || n_cls <= 0) {
        return -11;
    }

    std::lock_guard<std::mutex> lock(eng->mu);

    int64_t total_tokens_64 = 0;
    int64_t total_slots_64 = 0;
    for (int32_t s = 0; s < n_seqs; ++s) {
        if (seq_lens[s] <= 0 || seq_slot_counts[s] <= 0) {
            return -1;
        }
        total_tokens_64 += seq_lens[s];
        total_slots_64 += seq_slot_counts[s];
    }
    if (total_tokens_64 <= 0 || total_tokens_64 > eng->n_batch || total_tokens_64 > eng->n_ctx) {
        return -1;
    }
    if (total_slots_64 <= 0 || total_slots_64 > eng->n_outputs_max) {
        return -2;
    }

    for (int32_t c = 0; c < n_cls; ++c) {
        if (classifier_ids[c] < 0 || classifier_ids[c] >= eng->n_vocab) {
            return -5;
        }
    }

    const int32_t total_tokens = static_cast<int32_t>(total_tokens_64);
    const int32_t total_slots = static_cast<int32_t>(total_slots_64);

    llama_memory_clear(llama_get_memory(eng->ctx), true);

    llama_batch batch = llama_batch_init(total_tokens, 0, 1);
    std::vector<int32_t> slot_batch_indices;
    slot_batch_indices.reserve(static_cast<size_t>(total_slots));

    int32_t tok_off = 0;
    int32_t slot_off = 0;
    int32_t idx = 0;
    for (int32_t s = 0; s < n_seqs; ++s) {
        const int32_t L = seq_lens[s];
        const int32_t K = seq_slot_counts[s];
        std::vector<uint8_t> is_slot(static_cast<size_t>(L), 0);

        for (int32_t k = 0; k < K; ++k) {
            const int32_t sp = slot_positions[slot_off + k];
            if (sp < 0 || sp >= L) {
                llama_batch_free(batch);
                return -3;
            }
            is_slot[static_cast<size_t>(sp)] = 1;
            slot_batch_indices.push_back(idx + sp);
        }

        for (int32_t p = 0; p < L; ++p) {
            batch.token[idx] = token_ids[tok_off + p];
            batch.pos[idx] = p;
            batch.n_seq_id[idx] = 1;
            batch.seq_id[idx][0] = static_cast<llama_seq_id>(s);
            batch.logits[idx] = is_slot[static_cast<size_t>(p)] ? 1 : 0;
            idx++;
        }

        tok_off += L;
        slot_off += K;
    }
    batch.n_tokens = total_tokens;

    const int32_t rc = llama_decode(eng->ctx, batch);
    if (rc != 0) {
        llama_batch_free(batch);
        return 100 + rc;
    }

    for (size_t i = 0; i < slot_batch_indices.size(); ++i) {
        const float * row = llama_get_logits_ith(eng->ctx, slot_batch_indices[i]);
        if (!row) {
            llama_batch_free(batch);
            return -4;
        }
        float * dst = out_logits + i * static_cast<size_t>(n_cls);
        for (int32_t c = 0; c < n_cls; ++c) {
            dst[c] = row[classifier_ids[c]];
        }
    }

    llama_batch_free(batch);
    return 0;
}

void hs_engine_destroy(HsLlamaEngine * eng) {
    if (!eng) {
        return;
    }
    if (eng->ctx) {
        llama_free(eng->ctx);
    }
    if (eng->model) {
        llama_model_free(eng->model);
    }
    delete eng;
}

} // extern "C"
