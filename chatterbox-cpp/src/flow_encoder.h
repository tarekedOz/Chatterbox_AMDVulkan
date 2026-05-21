#pragma once

// Flow encoder — the input projection + UpsampleConformerEncoder that
// sits between T3's speech-token output and the CFM decoder.
//
// Three logical pieces, fused into one class for graph efficiency:
//
//   1. input_embedding             Embedding(6561, 512)
//   2. UpsampleConformerEncoder    6 conformer blocks @ 25 Hz
//                                  + 2x nearest-conv upsample
//                                  + 4 conformer blocks @ 50 Hz
//                                  + final LayerNorm
//   3. encoder_proj                Linear(512 -> 80)
//
// Plus, separately, the speaker affine layer that feeds the CFM decoder:
//
//   4. spk_embed_affine_layer      Linear(192 -> 80)
//
// Reference oracle: scripts/reference_flow_encoder.py. Surveys:
// docs/s3gen-survey.md (Component 3 + 4).
//
// Notable details:
//   - Conformer blocks are normalize_before, use_cnn_module=False,
//     macaron_style=False. No conv module — pure (norm -> rel-pos MHA ->
//     norm -> swish FF).
//   - Self-attention is RelPositionMultiHeadedAttention (Transformer-XL
//     style) over EspnetRelPositionalEncoding. The position embedding is
//     not learnable; the rel_shift trick is implemented in graph.
//   - pre_lookahead_layer is a small causal convolution block with a
//     residual; conv1 is right-padded by pre_lookahead_len=3, conv2 is
//     left-padded by 2.
//   - up_layer = nearest-neighbor 2x interpolate + left-pad 4 + conv k=5,
//     all stride 1.

#include "model.h"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

struct ggml_backend;
typedef ggml_backend* ggml_backend_t;
struct ggml_context;
struct ggml_tensor;

namespace chatterbox {

struct FlowEncoderConfig {
    int32_t vocab_size  = 6561;
    int32_t d_model     = 512;
    int32_t n_heads     = 8;
    int32_t d_head      = 64;            // d_model / n_heads
    int32_t ff_hidden   = 2048;
    int32_t n_blocks    = 6;
    int32_t n_up_blocks = 4;
    int32_t up_stride   = 2;
    int32_t pre_lookahead_len = 3;
    int32_t spk_in      = 192;
    int32_t d_out       = 80;
    float   ln_eps      = 1e-5f;          // encoder.embed / after_norm
    float   ln_eps_attn = 1e-12f;         // encoder_layer norm_mha / norm_ff
};

class FlowEncoder {
public:
    static std::unique_ptr<FlowEncoder> load(Model* s3gen_model);

    ~FlowEncoder();
    FlowEncoder(const FlowEncoder&) = delete;
    FlowEncoder& operator=(const FlowEncoder&) = delete;

    const FlowEncoderConfig& config() const { return cfg_; }

    // ---- Forward (full pipeline; encoder + encoder_proj) ----
    //
    // tokens:   T speech-token ids in [0, vocab_size).
    // T_in:     number of input tokens.
    // Returns the encoder_proj output: row-major (2T, 80) — note that the
    // upsample doubles the time dim from T to 2T.
    std::vector<float> forward(const std::vector<int32_t>& tokens,
                                int& out_T_out, int& out_d);

    // ---- Bisect-debug forward ----
    //
    // Run the full graph but read out every intermediate stage by name,
    // matching the dump names produced by reference_flow_encoder.py.
    // Result: row-major (T_at_stage, C_at_stage). Use stage_shape() to
    // recover (T, C) per stage.
    //
    // Stage names:
    //   after_input_embedding, after_embed, after_prelookahead,
    //   after_enc_block0, after_enc_block5,
    //   after_uplayer, after_upembed,
    //   after_upenc_block0, after_upenc_block3,
    //   after_afternorm, after_encoderproj
    std::vector<float> forward_with_stages(
        const std::vector<int32_t>& tokens,
        std::unordered_map<std::string, std::vector<float>>& stages_out,
        std::unordered_map<std::string, std::pair<int, int>>& stage_shapes);

    // ---- Speaker affine (separate; used by the CFM decoder, not the encoder) ----
    //
    // spk_192 is expected to already be L2-normalized (upstream does
    // F.normalize(emb, dim=1) right before calling this layer).
    std::vector<float> affine_speaker(const std::vector<float>& spk_192);

    struct Block {
        ggml_tensor* norm_mha_w = nullptr; ggml_tensor* norm_mha_b = nullptr;
        ggml_tensor* norm_ff_w  = nullptr; ggml_tensor* norm_ff_b  = nullptr;
        ggml_tensor* lin_q_w    = nullptr; ggml_tensor* lin_q_b    = nullptr;
        ggml_tensor* lin_k_w    = nullptr; ggml_tensor* lin_k_b    = nullptr;
        ggml_tensor* lin_v_w    = nullptr; ggml_tensor* lin_v_b    = nullptr;
        ggml_tensor* lin_out_w  = nullptr; ggml_tensor* lin_out_b  = nullptr;
        ggml_tensor* lin_pos_w  = nullptr;
        ggml_tensor* pos_bias_u = nullptr;     // (Hd, H)
        ggml_tensor* pos_bias_v = nullptr;
        ggml_tensor* ff_w1_w    = nullptr; ggml_tensor* ff_w1_b    = nullptr;
        ggml_tensor* ff_w2_w    = nullptr; ggml_tensor* ff_w2_b    = nullptr;
    };

private:
    FlowEncoder() = default;

    bool fetch_(Model* m);

    FlowEncoderConfig cfg_;
    Model*            model_   = nullptr;
    ggml_backend_t    backend_ = nullptr;

    // Top-level weights.
    ggml_tensor* input_embedding_w_ = nullptr;     // (D, vocab)
    ggml_tensor* spk_affine_w_      = nullptr;     // (192, 80)
    ggml_tensor* spk_affine_b_      = nullptr;     // (80,)
    ggml_tensor* encoder_proj_w_    = nullptr;     // (D, 80)
    ggml_tensor* encoder_proj_b_    = nullptr;     // (80,)

    // encoder.embed / up_embed (LinearNoSubsampling = Linear + LayerNorm).
    ggml_tensor* embed_lin_w_       = nullptr;
    ggml_tensor* embed_lin_b_       = nullptr;
    ggml_tensor* embed_ln_w_        = nullptr;
    ggml_tensor* embed_ln_b_        = nullptr;
    ggml_tensor* up_embed_lin_w_    = nullptr;
    ggml_tensor* up_embed_lin_b_    = nullptr;
    ggml_tensor* up_embed_ln_w_     = nullptr;
    ggml_tensor* up_embed_ln_b_     = nullptr;

    // encoder.pre_lookahead_layer.
    ggml_tensor* prelook_conv1_w_   = nullptr;     // (K=4, D, D)
    ggml_tensor* prelook_conv1_b_   = nullptr;
    ggml_tensor* prelook_conv2_w_   = nullptr;     // (K=3, D, D)
    ggml_tensor* prelook_conv2_b_   = nullptr;

    // encoder.up_layer.
    ggml_tensor* up_layer_conv_w_   = nullptr;     // (K=5, D, D)
    ggml_tensor* up_layer_conv_b_   = nullptr;

    // encoder.after_norm.
    ggml_tensor* after_norm_w_      = nullptr;
    ggml_tensor* after_norm_b_      = nullptr;

    std::vector<Block> encoders_;        // 6
    std::vector<Block> up_encoders_;     // 4
};

}  // namespace chatterbox
