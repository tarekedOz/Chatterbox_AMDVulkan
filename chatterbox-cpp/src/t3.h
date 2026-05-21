#pragma once

// T3 (autoregressive backbone) forward pass.
//
// Phase 1.E scope: a single forward step on a sequence of text token ids
// produces logits over the speech vocabulary at the LAST position.
// No conditioning (no speaker embedding or cond_prompt_speech_tokens),
// no KV cache, no sampling loop. Those layers on later.
//
// Validation oracle: scripts/reference_t3_forward.py (NumPy fp32) writes
// tests/t3_reference.bin with the expected logits for a fixed input.

#include "model.h"
#include "sampler.h"

#include <cstdint>
#include <memory>
#include <vector>

struct ggml_backend;
typedef ggml_backend* ggml_backend_t;
struct ggml_backend_buffer;
typedef ggml_backend_buffer* ggml_backend_buffer_t;
struct ggml_backend_sched;
typedef ggml_backend_sched* ggml_backend_sched_t;
struct ggml_context;

namespace chatterbox {

struct T3Config {
    int32_t n_layers      = 24;
    int32_t n_heads       = 16;
    int32_t embed_dim     = 1024;
    int32_t head_dim      = 64;
    int32_t ffn_dim       = 4096;
    int32_t context_len   = 8196;
    int32_t text_vocab    = 50276;
    int32_t speech_vocab  = 6563;
    int32_t speaker_emb_dim = 256;
    float   layer_norm_eps = 1e-5f;
};

class T3 {
public:
    // Construct from an already-loaded Model. Must have arch chatterbox_t3.
    // max_context bounds the KV cache; sequences longer than this will be
    // rejected at prefill / append time.
    static std::unique_ptr<T3> load(Model* model, int max_context = 2048);

    ~T3();
    T3(const T3&) = delete;
    T3& operator=(const T3&) = delete;

    // Reset the KV cache. Call before starting a new sequence.
    void reset_cache();
    int  n_past() const { return n_past_; }
    int  max_context() const { return max_context_; }

    // Prefill: full conditioning prefix + text tokens. Resets the cache.
    // Returns logits at the last position.
    //
    // Sequence layout (Turbo, matches upstream T3CondEnc.forward):
    //   [ cond_spkr (1 token from cond_enc.Linear(speaker_emb)),
    //     cond_prompt_speech (N_prompt tokens via speech_emb),
    //     text             (N_text tokens via text_emb) ]
    //
    // cond_prompt_speech_tokens may be empty — the chain still works,
    // it just means no audio-prompt conditioning is applied. Real
    // Chatterbox supplies ~375 prompt tokens from the S3 tokenizer.
    std::vector<float> prefill(
        const std::vector<float>& speaker_emb,
        const std::vector<int32_t>& cond_prompt_speech_tokens,
        const std::vector<int32_t>& text_tokens);

    // Extend the cache by N text tokens (via text_emb). Cache must be
    // populated via prefill first.
    std::vector<float> append_text(const std::vector<int32_t>& tokens);

    // Extend the cache by N speech tokens (via speech_emb). Used during
    // autoregressive generation: each newly sampled speech token is fed
    // back through here. Cache must be populated via prefill first.
    std::vector<float> append_speech(const std::vector<int32_t>& tokens);

    // Autoregressive speech-token generation. Assumes prefill() has been
    // called. Feeds `start_token` once, then repeatedly samples and feeds
    // back through append_speech until a token at or above `vocab_limit`
    // appears (EOS / OOV) or `max_tokens` is reached.
    // Returns the generated speech-token sequence (excludes start_token).
    struct GenParams {
        SamplingParams sampling{};
        int     max_tokens   = 1000;
        int32_t start_token  = 6561;   // upstream's speech BOS (vocab_size - 2)
        int32_t vocab_limit  = 6561;   // tokens >= this terminate generation
    };
    std::vector<int32_t> generate(const GenParams& params);

    const T3Config& config() const { return cfg_; }

private:
    T3() = default;

    struct Layer {
        ggml_tensor *ln1_w, *ln1_b;
        ggml_tensor *qkv_w, *qkv_b;
        ggml_tensor *out_w, *out_b;
        ggml_tensor *ln2_w, *ln2_b;
        ggml_tensor *fc_w,  *fc_b;
        ggml_tensor *proj_w, *proj_b;
    };

    T3Config cfg_;
    Model*   model_ = nullptr;

    // Weight handles (lifetime owned by model_->ctx_).
    ggml_tensor *text_emb_      = nullptr;
    ggml_tensor *speech_emb_    = nullptr;
    ggml_tensor *pos_embd_      = nullptr;
    ggml_tensor *cond_enc_w_    = nullptr;
    ggml_tensor *cond_enc_b_    = nullptr;
    ggml_tensor *out_norm_w_    = nullptr;
    ggml_tensor *out_norm_b_    = nullptr;
    ggml_tensor *speech_head_w_ = nullptr;
    ggml_tensor *speech_head_b_ = nullptr;
    std::vector<Layer> layers_;

    ggml_backend_t backend_ = nullptr;

    // Backend scheduler. Created once at load() and reused across every
    // forward_impl call (reset between graphs) — recreating it per
    // autoregressive step cost ~8 ms/step of pure setup overhead.
    ggml_backend_sched_t sched_ = nullptr;

    // Pre-transposed attention/FFN weights. The GGUF stores qkv/out/fc/
    // proj as (out, in); mul_mat needs (in, out). We transpose once at
    // load into this buffer so forward_impl skips a per-step
    // cont(transpose(.)) on every layer (~96 ops/step).
    ggml_context*         tw_ctx_    = nullptr;
    ggml_backend_buffer_t tw_buffer_ = nullptr;

    // KV cache. Allocated once at load(); persists across prefill/append calls.
    int                   max_context_   = 0;
    int                   n_past_        = 0;
    ggml_context*         cache_ctx_     = nullptr;
    ggml_tensor*          cache_k_       = nullptr;
    ggml_tensor*          cache_v_       = nullptr;
    ggml_backend_buffer_t cache_buffer_  = nullptr;

    // Internal forward.
    //   - has_cond_prefix=true emits cond_enc(speaker_emb) followed by
    //     speech_emb[cond_prompt_speech_tokens] before the main `tokens`.
    //   - has_cond_prefix=false ignores speaker_emb and prompt_tokens; it
    //     just embeds `tokens` via emb_table and runs the transformer.
    //   - emb_table is text_emb_ or speech_emb_ (the main-token table).
    // Updates n_past_ on success.
    std::vector<float> forward_impl(
        bool has_cond_prefix,
        const std::vector<float>& speaker_emb,
        const std::vector<int32_t>& cond_prompt_speech_tokens,
        const std::vector<int32_t>& tokens,
        ggml_tensor* emb_table);
};

}  // namespace chatterbox
