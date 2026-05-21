#pragma once

// S3 audio tokenizer encoder (AudioEncoderV2 from xingchensong/S3Tokenizer).
//
// log_mel(128, T_mel)
//   -> Conv1d(128 -> 1280, k=3, stride=2, pad=1) -> GELU(erf)
//   -> Conv1d(1280 -> 1280, k=3, stride=2, pad=1) -> GELU
//   -> permute to (T_tok, 1280)         where T_tok = T_mel / 4
//   -> 6 x ResidualAttentionBlock with FSMN-MHA + half-split RoPE
//   -> hidden (T_tok, 1280)
//
// Loaded from a chatterbox_s3gen GGUF; tensors live under
// `tokenizer.encoder.*`.
//
// This is the second of three pieces of the S3 tokenizer port; the
// quantizer (FSQ 3^8) comes next. See docs/s3-tokenizer-survey.md.

#include "model.h"

#include <cstdint>
#include <memory>
#include <vector>

struct ggml_backend;
typedef ggml_backend* ggml_backend_t;

namespace chatterbox {

struct S3EncConfig {
    int32_t n_mels    = 128;
    int32_t n_state   = 1280;
    int32_t n_head    = 20;
    int32_t head_dim  = 64;
    int32_t n_layer   = 6;
    int32_t fsmn_k    = 31;
    int32_t rope_max  = 2048;
    float   rope_base = 10000.0f;
    float   layer_norm_eps = 1e-5f;
};

class S3Encoder {
public:
    static std::unique_ptr<S3Encoder> load(Model* s3gen_model);

    ~S3Encoder();
    S3Encoder(const S3Encoder&) = delete;
    S3Encoder& operator=(const S3Encoder&) = delete;

    // log_mel input: row-major (n_mels, T_mel) — same layout that
    // MelExtractor::log_mel returns.
    // Output: (T_tok, n_state) row-major, T_tok = T_mel / 4.
    std::vector<float> forward(const std::vector<float>& log_mel,
                                int n_mels, int T_mel,
                                int& out_T_tok, int& out_n_state);

    const S3EncConfig& config() const { return cfg_; }

private:
    S3Encoder() = default;

    struct Block {
        ggml_tensor *attn_ln_w, *attn_ln_b;
        ggml_tensor *q_w, *q_b;
        ggml_tensor *k_w;           // no bias
        ggml_tensor *v_w, *v_b;
        ggml_tensor *out_w, *out_b;
        ggml_tensor *fsmn_w;        // depthwise (D, 1, K)
        ggml_tensor *mlp_ln_w, *mlp_ln_b;
        ggml_tensor *mlp_fc_w, *mlp_fc_b;
        ggml_tensor *mlp_proj_w, *mlp_proj_b;
    };

    S3EncConfig cfg_;
    Model*      model_ = nullptr;

    ggml_tensor *conv1_w_ = nullptr;
    ggml_tensor *conv1_b_ = nullptr;
    ggml_tensor *conv2_w_ = nullptr;
    ggml_tensor *conv2_b_ = nullptr;
    std::vector<Block> blocks_;

    ggml_backend_t backend_ = nullptr;
};

}  // namespace chatterbox
