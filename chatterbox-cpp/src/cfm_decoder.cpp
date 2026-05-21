#include "cfm_decoder.h"
#include "backend.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace chatterbox {

namespace {

constexpr size_t GRAPH_MAX_NODES = 262144;   // 12 mid + 4 transformers per stage * many ops

// ---- Activations / norms ----

// LayerNorm with gain + bias along ne[0].
ggml_tensor* layer_norm(ggml_context* ctx, ggml_tensor* x,
                         ggml_tensor* w, ggml_tensor* b, float eps) {
    ggml_tensor* n = ggml_norm(ctx, x, eps);
    return ggml_add(ctx, ggml_mul(ctx, n, w), b);
}

// Mish: x * tanh(softplus(x)).
ggml_tensor* mish(ggml_context* ctx, ggml_tensor* x) {
    return ggml_mul(ctx, x, ggml_tanh(ctx, ggml_softplus(ctx, x)));
}

// Add a 1D bias (ne[0]=C) to a 2D conv1d output with layout (T, C)
// (ne[0]=T, ne[1]=C). Reshape bias to (1, C) so broadcasting hits ne[1].
ggml_tensor* add_bias_TC(ggml_context* ctx, ggml_tensor* x, ggml_tensor* b,
                          int C) {
    return ggml_add(ctx, x, ggml_reshape_2d(ctx, b, 1, C));
}

// Compute sinusoidal positional embedding for a single scalar t.
// Returns a 1D vector of length d_model (=cfg.time_emb_in=320).
std::vector<float> sinusoidal_pos_emb(float t, int dim,
                                       float scale = 1000.0f) {
    std::vector<float> out(dim);
    const int half = dim / 2;
    const float log10000 = std::log(10000.0f);
    const float factor = log10000 / static_cast<float>(half - 1);
    for (int j = 0; j < half; ++j) {
        const float div = std::exp(-factor * static_cast<float>(j));
        const float a   = scale * t * div;
        out[j]        = std::sin(a);
        out[j + half] = std::cos(a);
    }
    return out;
}

// Left-pad a (T, C) tensor by `pad` zero rows along ne[0] (the time axis).
// Implements the left-padding required by CausalConv1d by concat with a
// zero-scaled view (same trick used in flow_encoder).
ggml_tensor* left_pad_TC(ggml_context* ctx, ggml_tensor* x_TC, int pad) {
    if (pad <= 0) return x_TC;
    const int T = static_cast<int>(x_TC->ne[0]);
    const int C = static_cast<int>(x_TC->ne[1]);
    (void) T;
    ggml_tensor* zero_view =
        ggml_view_2d(ctx, x_TC, pad, C,
                      /*nb1=*/x_TC->nb[1], /*offset=*/0);
    ggml_tensor* zero_pad =
        ggml_scale(ctx, ggml_cont(ctx, zero_view), 0.0f);
    return ggml_concat(ctx, zero_pad, x_TC, /*dim=*/0);
}

// CausalConv1d: pad left by K-1, then conv. Input/output layout (C, T)
// at the boundaries; this function does the (C, T) -> (T, C) -> conv ->
// (T, C_out) -> (C_out, T) conversion internally.
//
// We inline ggml_conv_1d's im2col + mul_mat decomposition so we can
// call ggml_mul_mat_set_prec(GGML_PREC_F32) on the matmul — required
// for Vulkan parity (fp16 accumulation across the resnet/mid stack
// otherwise drifts ~1.25e-2 per element by the first resnet output,
// then amplifies catastrophically through 12 mid layers).
ggml_tensor* causal_conv1d(ggml_context* ctx,
                             ggml_tensor* x_CT, int /*C*/, int /*T*/,
                             ggml_tensor* w, ggml_tensor* b,
                             int K, int C_out) {
    // (C, T) -> (T, C) for conv1d input.
    ggml_tensor* x_TC = ggml_cont(ctx, ggml_transpose(ctx, x_CT));
    // Left-pad with K-1 zero rows.
    ggml_tensor* padded = left_pad_TC(ctx, x_TC, K - 1);
    // im2col [IC*K, OL, N, ?]
    ggml_tensor* im2col = ggml_im2col(ctx, w, padded, /*s0=*/1, 0,
                                          /*p0=*/0, 0, /*d0=*/1, 0,
                                          false, GGML_TYPE_F16);
    // Match ggml_conv_1d's decomposition: mul_mat(im2col_2d, w_2d).
    //   im2col_2d ne=(IC*K, N*OL)
    //   w_2d      ne=(IC*K, OC)
    //   y_2d      ne=(N*OL, OC)
    ggml_tensor* im2col_2d = ggml_reshape_2d(ctx, im2col,
                                                  im2col->ne[0],
                                                  im2col->ne[1] * im2col->ne[2]);
    ggml_tensor* w_2d = ggml_reshape_2d(ctx, w,
                                            w->ne[0] * w->ne[1], w->ne[2]);
    ggml_tensor* y_2d = ggml_mul_mat(ctx, im2col_2d, w_2d);
    ggml_mul_mat_set_prec(y_2d, GGML_PREC_F32);
    // y_2d ne=(OL*N=OL, OC) for N=1 — already (T, C_out) row-major.
    ggml_tensor* y_TC = y_2d;
    if (b) y_TC = add_bias_TC(ctx, y_TC, b, C_out);
    return ggml_cont(ctx, ggml_transpose(ctx, y_TC));    // back to (C_out, T)
}

// Apply a 1x1 Conv1d as a per-position Linear. Input (C_in, T), output
// (C_out, T). We use mul_mat since 1x1 conv == per-position linear.
// Note: the weight in GGUF is (K=1, C_in, C_out) ggml ne. For 1x1 we can
// reshape to (C_in, C_out) and use mul_mat.
ggml_tensor* conv1x1(ggml_context* ctx, ggml_tensor* x_CT,
                       ggml_tensor* w, ggml_tensor* b,
                       int C_in, int C_out) {
    ggml_tensor* w_2d = ggml_reshape_2d(ctx, w, C_in, C_out);
    ggml_tensor* y    = ggml_mul_mat(ctx, w_2d, x_CT);     // (C_out, T)
    ggml_mul_mat_set_prec(y, GGML_PREC_F32);
    if (b) y = ggml_add(ctx, y, b);
    return y;
}

// CausalBlock1D forward: x_CT (C, T) * mask (1, T) -> conv -> LN -> Mish -> *mask.
ggml_tensor* causal_block1d(ggml_context* ctx, ggml_tensor* x_CT,
                              ggml_tensor* mask_1T,
                              ggml_tensor* conv_w, ggml_tensor* conv_b,
                              ggml_tensor* ln_w,   ggml_tensor* ln_b,
                              int C_in, int C_out, float ln_eps) {
    (void) C_in;
    // Pre-mask
    ggml_tensor* h = ggml_mul(ctx, x_CT, mask_1T);     // broadcasts mask across C
    // Causal conv (K=3) + bias
    h = causal_conv1d(ctx, h, C_in, /*T=*/0, conv_w, conv_b, 3, C_out);
    // LayerNorm over ne[0]=C
    h = layer_norm(ctx, h, ln_w, ln_b, ln_eps);
    // Mish
    h = mish(ctx, h);
    // Post-mask
    return ggml_mul(ctx, h, mask_1T);
}

// CausalResnetBlock1D forward.
ggml_tensor* causal_resnet_block1d(ggml_context* ctx, ggml_tensor* x_CT,
                                     ggml_tensor* mask_1T,
                                     ggml_tensor* t_emb_1D,
                                     const CFMDecoder::ResnetWeights& W,
                                     int C_in, int C_out, float ln_eps) {
    // block1
    ggml_tensor* h = causal_block1d(ctx, x_CT, mask_1T,
                                      W.b1_conv_w, W.b1_conv_b,
                                      W.b1_ln_w,   W.b1_ln_b,
                                      C_in, C_out, ln_eps);
    // time MLP: mish(t_emb) -> Linear -> (C_out,). Then broadcast-add over T.
    ggml_tensor* tm = mish(ctx, t_emb_1D);                     // (D_time,)
    tm = ggml_add(ctx, ggml_mul_mat(ctx, W.mlp_w, tm), W.mlp_b); // (C_out,)
    // Reshape to (C_out, 1) for broadcast over T.
    tm = ggml_reshape_2d(ctx, tm, C_out, 1);
    h  = ggml_add(ctx, h, tm);

    // block2
    h = causal_block1d(ctx, h, mask_1T,
                        W.b2_conv_w, W.b2_conv_b,
                        W.b2_ln_w,   W.b2_ln_b,
                        C_out, C_out, ln_eps);

    // res_conv: 1x1 conv on (x * mask)
    ggml_tensor* sc_in = ggml_mul(ctx, x_CT, mask_1T);
    ggml_tensor* sc    = conv1x1(ctx, sc_in, W.res_w, W.res_b, C_in, C_out);
    return ggml_add(ctx, h, sc);
}

// BasicTransformerBlock forward.
//
// x_CT layout: (C=256, T). We rotate to (C, T) which matches the
// transformer convention used by flow_encoder. Inside attention we use
// the per-head reshape pattern from s3enc.cpp/flow_encoder.cpp.
//
// attn_mask is currently UNUSED — we feed all-1s masks at our scale and
// the original add_optional_chunk_mask reduces to an all-zeros additive
// bias, which is a no-op.
ggml_tensor* basic_transformer_block(ggml_context* ctx, ggml_tensor* x_CT,
                                      const CFMDecoder::TransformerWeights& W,
                                      int C, int T, int H, int Hd,
                                      int attn_inner, int ff_hidden,
                                      float ln_eps) {
    (void) attn_inner;
    // ---- Self-attn branch ----
    ggml_tensor* h = layer_norm(ctx, x_CT, W.norm1_w, W.norm1_b, ln_eps);
    // Q, K, V projections (no bias). PREC_F32 — see causal_conv1d note.
    ggml_tensor* q = ggml_mul_mat(ctx, W.to_q_w, h);    // (attn_inner=512, T)
    ggml_tensor* k = ggml_mul_mat(ctx, W.to_k_w, h);
    ggml_tensor* v = ggml_mul_mat(ctx, W.to_v_w, h);
    ggml_mul_mat_set_prec(q, GGML_PREC_F32);
    ggml_mul_mat_set_prec(k, GGML_PREC_F32);
    ggml_mul_mat_set_prec(v, GGML_PREC_F32);
    // Reshape per-head: (Hd, H, T).
    ggml_tensor* qh = ggml_reshape_3d(ctx, q, Hd, H, T);
    ggml_tensor* kh = ggml_reshape_3d(ctx, k, Hd, H, T);
    ggml_tensor* vh = ggml_reshape_3d(ctx, v, Hd, H, T);
    // Permute Q, K to (Hd, T, H) for the matmul-as-attention pattern.
    ggml_tensor* Qp = ggml_cont(ctx, ggml_permute(ctx, qh, 0, 2, 1, 3));
    ggml_tensor* Kp = ggml_cont(ctx, ggml_permute(ctx, kh, 0, 2, 1, 3));
    // V_trans for attn @ V matmul: (T, Hd, H).
    ggml_tensor* V_trans = ggml_cont_3d(ctx,
        ggml_permute(ctx, vh, 1, 2, 0, 3), T, Hd, H);
    // Scores = K^T @ Q (per head) -> (T_k, T_q, H), scale by 1/sqrt(Hd).
    ggml_tensor* scores = ggml_mul_mat(ctx, Kp, Qp);
    ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
    scores = ggml_scale(ctx, scores, 1.0f / std::sqrt(static_cast<float>(Hd)));
    // Softmax over keys (ne[0]). No mask add (all-1s mask -> zero bias).
    scores = ggml_soft_max(ctx, scores);
    // Attention output = V_trans @ scores -> (Hd, T, H).
    ggml_tensor* attn = ggml_mul_mat(ctx, V_trans, scores);
    ggml_mul_mat_set_prec(attn, GGML_PREC_F32);
    // Permute back to (Hd, H, T) -> flatten heads to (C=512, T).
    ggml_tensor* attn_p = ggml_cont(ctx, ggml_permute(ctx, attn, 0, 2, 1, 3));
    ggml_tensor* attn_flat = ggml_reshape_2d(ctx, attn_p, H * Hd, T);
    // to_out Linear (attn_inner -> C) with bias.
    ggml_tensor* a_out_mm = ggml_mul_mat(ctx, W.to_out_w, attn_flat);
    ggml_mul_mat_set_prec(a_out_mm, GGML_PREC_F32);
    ggml_tensor* a_out = ggml_add(ctx, a_out_mm, W.to_out_b);
    x_CT = ggml_add(ctx, x_CT, a_out);

    // ---- Feed-forward branch ----
    h = layer_norm(ctx, x_CT, W.norm3_w, W.norm3_b, ln_eps);
    // ff.net.0 = GELU(dim, inner) — Linear + (exact erf) GELU.
    ggml_tensor* ff_p_mm = ggml_mul_mat(ctx, W.ff_p_w, h);
    ggml_mul_mat_set_prec(ff_p_mm, GGML_PREC_F32);
    h = ggml_add(ctx, ff_p_mm, W.ff_p_b);                            // (1024, T)
    h = ggml_gelu_erf(ctx, h);
    ggml_tensor* ff_2_mm = ggml_mul_mat(ctx, W.ff_2_w, h);
    ggml_mul_mat_set_prec(ff_2_mm, GGML_PREC_F32);
    h = ggml_add(ctx, ff_2_mm, W.ff_2_b);                            // (256, T)
    (void) ff_hidden;
    return ggml_add(ctx, x_CT, h);
}

ggml_tensor* transformer_stack(ggml_context* ctx, ggml_tensor* x_CT,
                                 const std::vector<CFMDecoder::TransformerWeights>& Ws,
                                 int C, int T, int H, int Hd,
                                 int attn_inner, int ff_hidden,
                                 float ln_eps) {
    for (const auto& W : Ws) {
        x_CT = basic_transformer_block(ctx, x_CT, W, C, T, H, Hd,
                                          attn_inner, ff_hidden, ln_eps);
    }
    return x_CT;
}

}  // namespace


// ----------------------------------------------------------------------------
// Loader
// ----------------------------------------------------------------------------

CFMDecoder::~CFMDecoder() {
    // backend_ is a process-wide singleton (see backend.h); do NOT free.
}

bool CFMDecoder::fetch_(Model* m) {
    auto get = [&](const std::string& name) -> ggml_tensor* {
        ggml_tensor* t = m->find_tensor(name);
        if (!t) std::fprintf(stderr, "CFMDecoder::load: missing %s\n",
                              name.c_str());
        return t;
    };

    const std::string root = "flow.decoder.estimator.";

    time_mlp_l1_w_ = get(root + "time_mlp.linear_1.weight");
    time_mlp_l1_b_ = get(root + "time_mlp.linear_1.bias");
    time_mlp_l2_w_ = get(root + "time_mlp.linear_2.weight");
    time_mlp_l2_b_ = get(root + "time_mlp.linear_2.bias");
    time_mixer_w_  = get(root + "time_embed_mixer.weight");

    auto fetch_resnet = [&](ResnetWeights& W, const std::string& p) {
        W.b1_conv_w = get(p + ".block1.block.0.weight");
        W.b1_conv_b = get(p + ".block1.block.0.bias");
        W.b1_ln_w   = get(p + ".block1.block.2.weight");
        W.b1_ln_b   = get(p + ".block1.block.2.bias");
        W.b2_conv_w = get(p + ".block2.block.0.weight");
        W.b2_conv_b = get(p + ".block2.block.0.bias");
        W.b2_ln_w   = get(p + ".block2.block.2.weight");
        W.b2_ln_b   = get(p + ".block2.block.2.bias");
        W.mlp_w     = get(p + ".mlp.1.weight");
        W.mlp_b     = get(p + ".mlp.1.bias");
        W.res_w     = get(p + ".res_conv.weight");
        W.res_b     = get(p + ".res_conv.bias");
        return W.b1_conv_w && W.mlp_w && W.res_w;
    };

    auto fetch_transformer = [&](TransformerWeights& W, const std::string& p) {
        W.norm1_w  = get(p + ".norm1.weight");
        W.norm1_b  = get(p + ".norm1.bias");
        W.norm3_w  = get(p + ".norm3.weight");
        W.norm3_b  = get(p + ".norm3.bias");
        W.to_q_w   = get(p + ".attn1.to_q.weight");
        W.to_k_w   = get(p + ".attn1.to_k.weight");
        W.to_v_w   = get(p + ".attn1.to_v.weight");
        W.to_out_w = get(p + ".attn1.to_out.0.weight");
        W.to_out_b = get(p + ".attn1.to_out.0.bias");
        W.ff_p_w   = get(p + ".ff.net.0.proj.weight");
        W.ff_p_b   = get(p + ".ff.net.0.proj.bias");
        W.ff_2_w   = get(p + ".ff.net.2.weight");
        W.ff_2_b   = get(p + ".ff.net.2.bias");
        return W.norm1_w && W.to_q_w && W.ff_p_w;
    };

    auto fetch_stage = [&](StageWeights& S, const std::string& root,
                            int idx) {
        char nm[160];
        std::snprintf(nm, sizeof(nm), "%s.%d", root.c_str(), idx);
        const std::string p(nm);
        if (!fetch_resnet(S.resnet, p + ".0")) return false;
        S.transformers.resize(cfg_.n_transformers);
        for (int t = 0; t < cfg_.n_transformers; ++t) {
            char tnm[200];
            std::snprintf(tnm, sizeof(tnm), "%s.1.%d", p.c_str(), t);
            if (!fetch_transformer(S.transformers[t], std::string(tnm))) {
                return false;
            }
        }
        S.conv_w = get(p + ".2.weight");
        S.conv_b = get(p + ".2.bias");
        return S.conv_w && S.conv_b;
    };

    if (!fetch_stage(down_, root + "down_blocks", 0)) return false;
    mid_.resize(cfg_.n_mid_blocks);
    for (int i = 0; i < cfg_.n_mid_blocks; ++i) {
        // Mid blocks don't have a .2 conv (just resnet + transformers).
        char nm[160];
        std::snprintf(nm, sizeof(nm), "%smid_blocks.%d", root.c_str(), i);
        const std::string p(nm);
        if (!fetch_resnet(mid_[i].resnet, p + ".0")) return false;
        mid_[i].transformers.resize(cfg_.n_transformers);
        for (int t = 0; t < cfg_.n_transformers; ++t) {
            char tnm[200];
            std::snprintf(tnm, sizeof(tnm), "%s.1.%d", p.c_str(), t);
            if (!fetch_transformer(mid_[i].transformers[t],
                                     std::string(tnm))) return false;
        }
        mid_[i].conv_w = nullptr;     // no post-stack conv
        mid_[i].conv_b = nullptr;
    }
    if (!fetch_stage(up_, root + "up_blocks", 0)) return false;

    fb_conv_w_ = get(root + "final_block.block.0.weight");
    fb_conv_b_ = get(root + "final_block.block.0.bias");
    fb_ln_w_   = get(root + "final_block.block.2.weight");
    fb_ln_b_   = get(root + "final_block.block.2.bias");
    fp_w_      = get(root + "final_proj.weight");
    fp_b_      = get(root + "final_proj.bias");

    return time_mlp_l1_w_ && time_mixer_w_ && fb_conv_w_ && fp_w_;
}

std::unique_ptr<CFMDecoder> CFMDecoder::load(Model* model) {
    if (!model) return nullptr;
    if (model->arch != "chatterbox_s3gen") {
        std::fprintf(stderr,
                     "CFMDecoder::load: expected chatterbox_s3gen arch, got %s\n",
                     model->arch.c_str());
        return nullptr;
    }
    auto d = std::unique_ptr<CFMDecoder>(new CFMDecoder());
    d->model_ = model;
    if (!d->fetch_(model)) return nullptr;

    d->backend_ = chatterbox::default_backend();
    if (!d->backend_) {
        std::fprintf(stderr, "CFMDecoder::load: backend init failed\n");
        return nullptr;
    }
    std::printf("CFMDecoder::load: ok (%d mid blocks)\n",
                d->cfg_.n_mid_blocks);
    return d;
}


// ----------------------------------------------------------------------------
// Estimator forward (with stages)
// ----------------------------------------------------------------------------

std::vector<float> CFMDecoder::estimator_forward_with_stages(
        const std::vector<float>& x,
        const std::vector<float>& mask,
        const std::vector<float>& mu,
        float t,
        const std::vector<float>& spks,
        const std::vector<float>& cond,
        float r, int T_mel,
        std::unordered_map<std::string, std::vector<float>>& stages_out,
        std::unordered_map<std::string, std::pair<int, int>>& stage_shapes) {
    stages_out.clear();
    stage_shapes.clear();
    if (T_mel <= 0) return {};

    const int C_in  = cfg_.in_channels;     // 320
    const int C_mid = cfg_.mid_channels;    // 256
    const int C_out = cfg_.out_channels;    // 80
    const int H     = cfg_.n_heads;
    const int Hd    = cfg_.head_dim;
    const int D_t   = cfg_.time_emb_dim;    // 1024

    // Host-side: compute t_emb and r_emb (sinusoidal). The TimestepEmbedding
    // MLP and the meanflow time_embed_mixer live inside the graph.
    auto t_sin = sinusoidal_pos_emb(t, cfg_.time_emb_in);
    auto r_sin = sinusoidal_pos_emb(r, cfg_.time_emb_in);

    const size_t buf_size =
        ggml_tensor_overhead() * GRAPH_MAX_NODES
        + ggml_graph_overhead_custom(GRAPH_MAX_NODES, false);
    std::vector<uint8_t> buf(buf_size);
    ggml_init_params params{};
    params.mem_size   = buf.size();
    params.mem_buffer = buf.data();
    params.no_alloc   = true;
    ggml_context* ctx = ggml_init(params);
    if (!ctx) return {};
    ggml_cgraph* gf = ggml_new_graph_custom(ctx, GRAPH_MAX_NODES, false);

    // ---- Inputs (all in (C, T) ggml layout) ----
    ggml_tensor* x_in    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, C_out, T_mel);
    ggml_tensor* mu_in   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, C_out, T_mel);
    ggml_tensor* cond_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, C_out, T_mel);
    ggml_tensor* spks_in = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, C_out);
    ggml_tensor* mask_in = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, T_mel);
    ggml_tensor* tsin_in = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, cfg_.time_emb_in);
    ggml_tensor* rsin_in = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, cfg_.time_emb_in);
    for (auto* tt : {x_in, mu_in, cond_in, spks_in, mask_in, tsin_in, rsin_in}) {
        ggml_set_input(tt);
    }
    ggml_set_name(x_in, "x_in");
    ggml_set_name(mu_in, "mu_in");
    ggml_set_name(cond_in, "cond_in");
    ggml_set_name(spks_in, "spks_in");
    ggml_set_name(mask_in, "mask_in");
    ggml_set_name(tsin_in, "tsin_in");
    ggml_set_name(rsin_in, "rsin_in");

    // Reshape mask to (1, T) so broadcast hits ne[1].
    ggml_tensor* mask_1T = ggml_reshape_2d(ctx, mask_in, 1, T_mel);

    std::vector<std::pair<std::string, ggml_tensor*>> stage_tensors;
    auto mark = [&](const char* name, ggml_tensor* tt) {
        ggml_set_name(tt, name);
        ggml_set_output(tt);
        stage_tensors.emplace_back(name, tt);
    };

    // ---- Time embedding ----
    auto time_mlp = [&](ggml_tensor* s) {
        ggml_tensor* h = ggml_add(ctx,
            ggml_mul_mat(ctx, time_mlp_l1_w_, s), time_mlp_l1_b_);
        h = ggml_silu(ctx, h);
        return ggml_add(ctx,
            ggml_mul_mat(ctx, time_mlp_l2_w_, h), time_mlp_l2_b_);
    };
    ggml_tensor* t_emb = time_mlp(tsin_in);
    mark("after_t_emb", t_emb);
    ggml_tensor* r_emb = time_mlp(rsin_in);
    mark("after_r_emb", r_emb);

    // concat([t_emb, r_emb]) (2048,) -> Linear(2048 -> 1024, no bias).
    ggml_tensor* tr = ggml_concat(ctx, t_emb, r_emb, /*dim=*/0);    // (2048,)
    ggml_tensor* t_mix = ggml_mul_mat(ctx, time_mixer_w_, tr);       // (1024,)
    mark("after_time_mixer", t_mix);

    // ---- Pack channels: [x, mu, spks_broadcast, cond] -> (320, T) ----
    // spks (80,) -> (80, 1) reshape, then ggml_repeat to broadcast across T.
    ggml_tensor* spks_CT = ggml_reshape_2d(ctx, spks_in, C_out, 1);
    ggml_tensor* spks_template =
        ggml_new_tensor_2d(ctx, GGML_TYPE_F32, C_out, T_mel);
    ggml_tensor* spks_BT = ggml_repeat(ctx, spks_CT, spks_template);

    // ggml_concat with dim=0 stacks along ne[0] (the channel axis), which
    // matches the upstream pack([..], "b * t") behavior on the channel dim.
    ggml_tensor* pack1  = ggml_concat(ctx, x_in,    mu_in,    /*dim=*/0);   // (160, T)
    ggml_tensor* pack2  = ggml_concat(ctx, pack1,   spks_BT,  /*dim=*/0);   // (240, T)
    ggml_tensor* packed = ggml_concat(ctx, pack2,   cond_in,  /*dim=*/0);   // (320, T)
    mark("after_pack", packed);

    // ---- Down block 0 ----
    // IMPORTANT: skip connection is captured AFTER the transformer stack
    // but BEFORE the post-stack conv (matches upstream's
    // `hiddens.append(x)` placement between transformer and downsample).
    ggml_tensor* h_t = causal_resnet_block1d(ctx, packed, mask_1T, t_mix,
                                                down_.resnet,
                                                C_in, C_mid, cfg_.ln_eps);
    mark("after_down0_resnet", h_t);
    h_t = transformer_stack(ctx, h_t, down_.transformers,
                              C_mid, T_mel, H, Hd,
                              cfg_.attn_inner, cfg_.ff_hidden,
                              cfg_.ln_eps);
    mark("after_down0_transformer", h_t);
    ggml_tensor* skip = h_t;  // skip captured here, before the conv
    {
        ggml_tensor* xm = ggml_mul(ctx, h_t, mask_1T);
        h_t = causal_conv1d(ctx, xm, C_mid, T_mel,
                              down_.conv_w, down_.conv_b, 3, C_mid);
        mark("after_down0_downsample", h_t);
    }

    // ---- Mid blocks ----
    for (int i = 0; i < cfg_.n_mid_blocks; ++i) {
        h_t = causal_resnet_block1d(ctx, h_t, mask_1T, t_mix,
                                       mid_[i].resnet,
                                       C_mid, C_mid, cfg_.ln_eps);
        h_t = transformer_stack(ctx, h_t, mid_[i].transformers,
                                  C_mid, T_mel, H, Hd,
                                  cfg_.attn_inner, cfg_.ff_hidden,
                                  cfg_.ln_eps);
        if (i == 0)                       mark("after_mid0",  h_t);
        if (i == cfg_.n_mid_blocks - 1)   mark("after_mid11", h_t);
    }

    // ---- Up block 0 ----
    // Concat with skip along ne[0]: (256 + 256, T) = (512, T).
    ggml_tensor* h_skip = ggml_concat(ctx, h_t, skip, /*dim=*/0);
    h_t = causal_resnet_block1d(ctx, h_skip, mask_1T, t_mix,
                                   up_.resnet,
                                   cfg_.skip_up_in, C_mid, cfg_.ln_eps);
    h_t = transformer_stack(ctx, h_t, up_.transformers,
                              C_mid, T_mel, H, Hd,
                              cfg_.attn_inner, cfg_.ff_hidden,
                              cfg_.ln_eps);
    mark("after_up0_transformer", h_t);
    {
        ggml_tensor* xm = ggml_mul(ctx, h_t, mask_1T);
        h_t = causal_conv1d(ctx, xm, C_mid, T_mel,
                              up_.conv_w, up_.conv_b, 3, C_mid);
        mark("after_up0_upsample", h_t);
    }

    // ---- final_block + final_proj ----
    h_t = causal_block1d(ctx, h_t, mask_1T,
                          fb_conv_w_, fb_conv_b_,
                          fb_ln_w_,   fb_ln_b_,
                          C_mid, C_mid, cfg_.ln_eps);
    mark("after_final_block", h_t);
    // final_proj: 1x1 conv (256 -> 80) on (h_t * mask)
    ggml_tensor* xm = ggml_mul(ctx, h_t, mask_1T);
    h_t = conv1x1(ctx, xm, fp_w_, fp_b_, C_mid, C_out);
    h_t = ggml_mul(ctx, h_t, mask_1T);
    mark("after_final_proj", h_t);

    ggml_build_forward_expand(gf, h_t);

    // ---- Allocate + fill inputs ----
    ggml_backend_sched_t sched = chatterbox::make_sched(GRAPH_MAX_NODES);
    // Parity is held by per-matmul GGML_PREC_F32 inside causal_conv1d,
    // conv1x1, and basic_transformer_block — no CPU pin needed.
    //
    // CHATTERBOX_CFM_PIN_STAGE=<mark-name> re-enables the legacy pin
    // (route the named stage's output through the CPU backend) as a
    // fallback diagnostic if parity ever regresses.
    if (sched && chatterbox::is_vulkan_active()) {
        if (const char* pin_env = std::getenv("CHATTERBOX_CFM_PIN_STAGE")) {
            ggml_tensor* pin = nullptr;
            const std::string pin_name = pin_env;
            for (auto& kv : stage_tensors) {
                if (kv.first == pin_name) { pin = kv.second; break; }
            }
            if (pin) {
                ggml_backend_sched_set_tensor_backend(
                    sched, pin, chatterbox::cpu_backend());
            }
        }
    }
    if (!sched || !ggml_backend_sched_alloc_graph(sched, gf)) {
        std::fprintf(stderr,
                     "CFMDecoder: galloc_alloc_graph failed\n");
        if (sched) ggml_backend_sched_free(sched);
        ggml_free(ctx);
        return {};
    }
    ggml_backend_tensor_set(x_in,    x.data(),    0, x.size()    * sizeof(float));
    ggml_backend_tensor_set(mu_in,   mu.data(),   0, mu.size()   * sizeof(float));
    ggml_backend_tensor_set(cond_in, cond.data(), 0, cond.size() * sizeof(float));
    ggml_backend_tensor_set(spks_in, spks.data(), 0, spks.size() * sizeof(float));
    ggml_backend_tensor_set(mask_in, mask.data(), 0, mask.size() * sizeof(float));
    ggml_backend_tensor_set(tsin_in, t_sin.data(), 0, t_sin.size() * sizeof(float));
    ggml_backend_tensor_set(rsin_in, r_sin.data(), 0, r_sin.size() * sizeof(float));

    if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr,
                     "CFMDecoder: graph_compute failed\n");
        if (sched) ggml_backend_sched_free(sched);
        ggml_free(ctx);
        return {};
    }

    for (const auto& [name, tt] : stage_tensors) {
        const int C_st = static_cast<int>(tt->ne[0]);
        const int T_st = static_cast<int>(tt->ne[1]);
        std::vector<float> out_st(static_cast<size_t>(C_st) *
                                   std::max(T_st, 1));
        ggml_backend_tensor_get(tt, out_st.data(), 0,
                                  out_st.size() * sizeof(float));
        stages_out[name] = std::move(out_st);
        stage_shapes[name] = {T_st, C_st};
    }

    auto final_out = stages_out.at("after_final_proj");
    if (sched) ggml_backend_sched_free(sched);
    ggml_free(ctx);
    return final_out;
}

std::vector<float> CFMDecoder::estimator_forward(
        const std::vector<float>& x,
        const std::vector<float>& mask,
        const std::vector<float>& mu,
        float t,
        const std::vector<float>& spks,
        const std::vector<float>& cond,
        float r, int T_mel) {
    std::unordered_map<std::string, std::vector<float>> stages;
    std::unordered_map<std::string, std::pair<int, int>> shapes;
    return estimator_forward_with_stages(x, mask, mu, t, spks, cond, r,
                                            T_mel, stages, shapes);
}


// ----------------------------------------------------------------------------
// Meanflow solver
// ----------------------------------------------------------------------------

std::vector<float> CFMDecoder::solve_meanflow(
        const std::vector<float>& z,
        const std::vector<float>& mu,
        const std::vector<float>& mask,
        const std::vector<float>& spks,
        const std::vector<float>& cond,
        int T_mel, int n_timesteps) {
    if (n_timesteps <= 0) return z;

    std::vector<float> x = z;
    for (int i = 0; i < n_timesteps; ++i) {
        const float t = static_cast<float>(i)     / n_timesteps;
        const float r = static_cast<float>(i + 1) / n_timesteps;
        const float dt = r - t;
        auto dxdt = estimator_forward(x, mask, mu, t, spks, cond, r, T_mel);
        if (dxdt.size() != x.size()) {
            std::fprintf(stderr,
                         "CFMDecoder::solve_meanflow: dxdt/x size mismatch\n");
            return {};
        }
        for (size_t j = 0; j < x.size(); ++j) {
            x[j] = x[j] + dt * dxdt[j];
        }
    }
    return x;
}

}  // namespace chatterbox
