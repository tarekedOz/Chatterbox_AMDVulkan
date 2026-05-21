#pragma once

// CFM decoder — the UNet noise predictor (ConditionalDecoder) plus the
// meanflow Euler solver (CausalConditionalCFM.basic_euler).
//
// Lives entirely inside the flow.decoder.* prefix of a chatterbox_s3gen
// GGUF. Mirrors upstream chatterbox/models/s3gen/{decoder.py,
// flow_matching.py}.
//
// Turbo configuration (channels=[256], n_blocks=4, num_mid_blocks=12,
// meanflow=True): a single down-stack, 12 mid blocks, a single
// up-stack with a skip connection. No actual time-domain sub/upsampling
// (the would-be downsamplers/upsamplers become plain CausalConv1d k=3
// because is_last==True on the only stage).
//
// Reference oracle: scripts/reference_cfm_decoder.py.
// Survey: docs/s3gen-survey.md Component 5.

#include "model.h"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

struct ggml_backend;
typedef ggml_backend* ggml_backend_t;
struct ggml_context;
struct ggml_tensor;

namespace chatterbox {

struct CFMDecoderConfig {
    int32_t in_channels        = 320;     // x(80) + mu(80) + spks(80) + cond(80)
    int32_t out_channels       = 80;
    int32_t mid_channels       = 256;
    int32_t time_emb_in        = 320;     // SinusoidalPosEmb dim (= in_channels)
    int32_t time_emb_dim       = 1024;    // mid_channels * 4
    int32_t skip_up_in         = 512;     // mid_channels * 2 (concat with skip)
    int32_t n_heads            = 8;
    int32_t head_dim           = 64;
    int32_t attn_inner         = 512;     // n_heads * head_dim
    int32_t ff_hidden          = 1024;    // mid_channels * 4
    int32_t n_transformers     = 4;       // per stage
    int32_t n_mid_blocks       = 12;
    int32_t n_timesteps        = 2;       // meanflow default
    float   ln_eps             = 1e-5f;
};

class CFMDecoder {
public:
    static std::unique_ptr<CFMDecoder> load(Model* s3gen_model);

    ~CFMDecoder();
    CFMDecoder(const CFMDecoder&) = delete;
    CFMDecoder& operator=(const CFMDecoder&) = delete;

    const CFMDecoderConfig& config() const { return cfg_; }

    // ---- Single-step estimator forward ----
    //
    // x:    (80, T) row-major  (noised mel)
    // mask: (T,)               (1s where valid, 0s where padded)
    // mu:   (80, T)
    // t:    scalar
    // spks: (80,)               (after spk_embed_affine_layer)
    // cond: (80, T)
    // r:    scalar (meanflow end time)
    //
    // Returns dxdt (80, T) row-major.
    std::vector<float> estimator_forward(const std::vector<float>& x,
                                          const std::vector<float>& mask,
                                          const std::vector<float>& mu,
                                          float t,
                                          const std::vector<float>& spks,
                                          const std::vector<float>& cond,
                                          float r, int T_mel);

    // Same but also returns intermediate stage tensors keyed by name,
    // matching the dump names produced by reference_cfm_decoder.py.
    std::vector<float> estimator_forward_with_stages(
        const std::vector<float>& x,
        const std::vector<float>& mask,
        const std::vector<float>& mu,
        float t,
        const std::vector<float>& spks,
        const std::vector<float>& cond,
        float r, int T_mel,
        std::unordered_map<std::string, std::vector<float>>& stages_out,
        std::unordered_map<std::string, std::pair<int, int>>& stage_shapes);

    // ---- Meanflow solver (basic_euler) ----
    //
    // z: (80, T) initial noise (provided by the caller for deterministic
    //    parity — numpy reference uses the same z).
    // Returns the final mel (80, T).
    std::vector<float> solve_meanflow(const std::vector<float>& z,
                                       const std::vector<float>& mu,
                                       const std::vector<float>& mask,
                                       const std::vector<float>& spks,
                                       const std::vector<float>& cond,
                                       int T_mel, int n_timesteps = 2);

    struct ResnetWeights {
        ggml_tensor* b1_conv_w = nullptr; ggml_tensor* b1_conv_b = nullptr;
        ggml_tensor* b1_ln_w   = nullptr; ggml_tensor* b1_ln_b   = nullptr;
        ggml_tensor* b2_conv_w = nullptr; ggml_tensor* b2_conv_b = nullptr;
        ggml_tensor* b2_ln_w   = nullptr; ggml_tensor* b2_ln_b   = nullptr;
        ggml_tensor* mlp_w     = nullptr; ggml_tensor* mlp_b     = nullptr;
        ggml_tensor* res_w     = nullptr; ggml_tensor* res_b     = nullptr;
    };
    struct TransformerWeights {
        ggml_tensor* norm1_w = nullptr; ggml_tensor* norm1_b = nullptr;
        ggml_tensor* norm3_w = nullptr; ggml_tensor* norm3_b = nullptr;
        ggml_tensor* to_q_w  = nullptr;                      // no bias
        ggml_tensor* to_k_w  = nullptr;
        ggml_tensor* to_v_w  = nullptr;
        ggml_tensor* to_out_w= nullptr; ggml_tensor* to_out_b = nullptr;
        ggml_tensor* ff_p_w  = nullptr; ggml_tensor* ff_p_b  = nullptr;
        ggml_tensor* ff_2_w  = nullptr; ggml_tensor* ff_2_b  = nullptr;
    };
    struct StageWeights {
        ResnetWeights resnet;
        std::vector<TransformerWeights> transformers;
        ggml_tensor* conv_w = nullptr;                       // post-stack conv
        ggml_tensor* conv_b = nullptr;
    };

private:
    CFMDecoder() = default;

    bool fetch_(Model* m);

    CFMDecoderConfig cfg_;
    Model*           model_   = nullptr;
    ggml_backend_t   backend_ = nullptr;

    // Top-level
    ggml_tensor* time_mlp_l1_w_ = nullptr; ggml_tensor* time_mlp_l1_b_ = nullptr;
    ggml_tensor* time_mlp_l2_w_ = nullptr; ggml_tensor* time_mlp_l2_b_ = nullptr;
    ggml_tensor* time_mixer_w_  = nullptr;     // Linear(2048 -> 1024) no bias

    StageWeights              down_;
    std::vector<StageWeights> mid_;            // 12
    StageWeights              up_;

    // Final block + projection
    ggml_tensor* fb_conv_w_  = nullptr; ggml_tensor* fb_conv_b_  = nullptr;
    ggml_tensor* fb_ln_w_    = nullptr; ggml_tensor* fb_ln_b_    = nullptr;
    ggml_tensor* fp_w_       = nullptr; ggml_tensor* fp_b_       = nullptr;
};

}  // namespace chatterbox
