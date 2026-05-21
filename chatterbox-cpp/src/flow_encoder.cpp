#include "flow_encoder.h"
#include "backend.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace chatterbox {

namespace {

constexpr size_t GRAPH_MAX_NODES = 65536;
constexpr int    MAX_REL_LEN     = 5000;

// mul_mat with fp32 accumulator on Vulkan. Necessary across the
// 10-conformer stack: fp16 accumulation drifts the residual ~3e-2
// per element by encoder_proj, breaking parity. See cfm_decoder.cpp
// for the same pattern. No-op on CPU (set_prec only affects backends
// that look at op_params[0]).
ggml_tensor* mm_f32(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b) {
    ggml_tensor* y = ggml_mul_mat(ctx, a, b);
    ggml_mul_mat_set_prec(y, GGML_PREC_F32);
    return y;
}

// conv1d via inlined im2col + fp32-acc mul_mat. Drop-in for
// ggml_conv_1d when we need PREC_F32 on the underlying matmul.
// Returns the same shape as ggml_conv_1d (OL, OC, N).
ggml_tensor* conv_1d_f32(ggml_context* ctx,
                            ggml_tensor* w, ggml_tensor* x,
                            int s0, int p0, int d0) {
    ggml_tensor* im2col = ggml_im2col(ctx, w, x, s0, 0, p0, 0, d0, 0,
                                          false, GGML_TYPE_F16);
    // im2col ne=(IC*K, OL, N) — see ggml_conv_1d in ggml.c.
    ggml_tensor* im2col_2d = ggml_reshape_2d(ctx, im2col,
                                                  im2col->ne[0],
                                                  im2col->ne[1] * im2col->ne[2]);
    ggml_tensor* w_2d = ggml_reshape_2d(ctx, w,
                                            w->ne[0] * w->ne[1], w->ne[2]);
    ggml_tensor* y = mm_f32(ctx, im2col_2d, w_2d);   // (OL*N, OC)
    return ggml_reshape_3d(ctx, y,
                              im2col->ne[1], w->ne[2], im2col->ne[2]);
}

// LayerNorm with gain + bias along ne[0]. Matches torch.nn.LayerNorm with
// normalize over the last dim (which is ne[0] in ggml convention).
ggml_tensor* layer_norm(ggml_context* ctx, ggml_tensor* x,
                         ggml_tensor* w, ggml_tensor* b, float eps) {
    ggml_tensor* n = ggml_norm(ctx, x, eps);
    return ggml_add(ctx, ggml_mul(ctx, n, w), b);
}

// Reshape a 1D bias (ne[0]=C) to (1, C) so it broadcasts across ne[0]
// when added to a (T, C) tensor (conv1d output layout).
ggml_tensor* bias_for_TC(ggml_context* ctx, ggml_tensor* b, int C) {
    return ggml_reshape_2d(ctx, b, 1, C);
}

// Compute the (D, 2T-1) slice of the EspnetRelPositionalEncoding table
// for a given length T. Layout: ne[0]=D (innermost), ne[1]=2T-1.
//
// The full table is constructed as:
//   For each position p in [0, max), compute sin/cos at d_model bins.
//   Layout: pe[l, d] for l in [0, 2*max-1)
//       l in [0, max):       pe[l, :] = positive_pe at offset (max - 1 - l)
//       l in [max, 2*max-1): pe[l, :] = negative_pe at offset (l - max + 1)
//
// The slice for size T returns pe[max-T+1 : max+T-1+1] = pe[max-T+1 : max+T].
std::vector<float> build_rel_pos_slice(int T, int d_model, int max_len) {
    const int L = 2 * T - 1;
    std::vector<float> out(static_cast<size_t>(L) * d_model);

    // The slice rows correspond to original-table indices [max-T+1, max+T).
    // For row k in [0, L), original index = max - T + 1 + k.
    // Whether positive or negative half depends on this index:
    //   if i  < max:   positive at offset (max-1-i)
    //   if i >= max:   negative at offset (i - max + 1)
    //
    // sin/cos bins: for 2j in [0, d_model, 2):
    //     div = exp(-log(10000) * 2j / d_model)
    //     out[k, 2j  ] = sin(pos * div)   (with pos signed)
    //     out[k, 2j+1] = cos(pos * div)
    // where pos = +offset for positive, -offset for negative.
    const int n_pairs = d_model / 2;
    std::vector<float> div(n_pairs);
    for (int j = 0; j < n_pairs; ++j) {
        div[j] = std::exp(-static_cast<float>(std::log(10000.0)) *
                            static_cast<float>(2 * j) /
                            static_cast<float>(d_model));
    }

    for (int k = 0; k < L; ++k) {
        // numpy's position_encoding slices pe[:, max_len-T : max_len+T-1].
        // The first row corresponds to table index max_len - T.
        const int i = max_len - T + k;
        int   offset;
        float sign;
        if (i < max_len) {
            offset = max_len - 1 - i;
            sign   = +1.0f;
        } else {
            offset = i - max_len + 1;
            sign   = -1.0f;
        }
        const float pos = sign * static_cast<float>(offset);
        float* row = out.data() + static_cast<size_t>(k) * d_model;
        for (int j = 0; j < n_pairs; ++j) {
            const float a = pos * div[j];
            row[2 * j    ] = std::sin(a);
            row[2 * j + 1] = std::cos(a);
        }
    }
    return out;
}

// Apply the ESPnet rel-shift trick.
//
// Input  bd: ne=(2T-1, T, H, 1)  (numpy (1, H, T, 2T-1))
// Output:    ne=(T,    T, H, 1)  (numpy (1, H, T, T))
//
// Algorithm:
//   1. Make a (1, T, H, 1) zero tensor by scaling-by-0 a slice of bd.
//   2. Concat along ne[0]: (2T, T, H, 1).
//   3. Reshape to (T, 2T, H, 1) — same memory, reinterpret strides.
//   4. View skipping the first row of ne[1] (offset T*sizeof(f32)) →
//      (T, 2T-1, H, 1), non-contiguous.
//   5. ggml_cont to compact to (T, 2T-1, H, 1) contig.
//   6. Reshape to (2T-1, T, H, 1) — same memory, reinterpret.
//   7. View first T of ne[0] → strided (T, T, H, 1).
//   8. ggml_cont to compact to (T, T, H, 1) contig.
ggml_tensor* rel_shift(ggml_context* ctx, ggml_tensor* bd, int T, int H) {
    // (1) Zero-pad by scaling a slice of bd by 0. The slice's element type
    // matches bd, which is what we need.
    const size_t bytes_f32 = sizeof(float);
    ggml_tensor* zero_slice =
        ggml_view_4d(ctx, bd,
                      /*ne0=*/1, /*ne1=*/T, /*ne2=*/H, /*ne3=*/1,
                      /*nb1=*/bd->nb[1], /*nb2=*/bd->nb[2], /*nb3=*/bd->nb[3],
                      /*offset=*/0);
    zero_slice = ggml_cont(ctx, zero_slice);                  // make contig
    zero_slice = ggml_scale(ctx, zero_slice, 0.0f);            // (1, T, H, 1)

    // (2) Concat along innermost dim ne[0]: (2T, T, H, 1).
    ggml_tensor* padded = ggml_concat(ctx, zero_slice, bd, /*dim=*/0);

    // (3) Reshape to (T, 2T, H, 1) — same memory.
    padded = ggml_reshape_4d(ctx, padded, T, 2 * T, H, 1);

    // (4) Skip the first row of ne[1]. New ne=(T, 2T-1, H, 1).
    const size_t nb1 = T * bytes_f32;
    const size_t nb2 = T * 2 * T * bytes_f32;
    const size_t nb3 = T * 2 * T * H * bytes_f32;
    ggml_tensor* skipped =
        ggml_view_4d(ctx, padded, T, 2 * T - 1, H, 1,
                      nb1, nb2, nb3, /*offset=*/T * bytes_f32);

    // (5) Compact.
    ggml_tensor* cont1 = ggml_cont(ctx, skipped);

    // (6) Reshape to (2T-1, T, H, 1).
    cont1 = ggml_reshape_4d(ctx, cont1, 2 * T - 1, T, H, 1);

    // (7) Take first T of ne[0].
    const size_t nb1b = (2 * T - 1) * bytes_f32;
    const size_t nb2b = T * (2 * T - 1) * bytes_f32;
    const size_t nb3b = T * (2 * T - 1) * H * bytes_f32;
    ggml_tensor* taken =
        ggml_view_4d(ctx, cont1, T, T, H, 1,
                      nb1b, nb2b, nb3b, /*offset=*/0);

    // (8) Compact.
    return ggml_cont(ctx, taken);
}

}  // namespace

FlowEncoder::~FlowEncoder() {
    // backend_ is a process-wide singleton (see backend.h); do NOT free.
}

bool FlowEncoder::fetch_(Model* m) {
    auto get = [&](const char* name) -> ggml_tensor* {
        ggml_tensor* t = m->find_tensor(name);
        if (!t) std::fprintf(stderr, "FlowEncoder::load: missing %s\n", name);
        return t;
    };

    input_embedding_w_   = get("flow.input_embedding.weight");
    spk_affine_w_        = get("flow.spk_embed_affine_layer.weight");
    spk_affine_b_        = get("flow.spk_embed_affine_layer.bias");
    encoder_proj_w_      = get("flow.encoder_proj.weight");
    encoder_proj_b_      = get("flow.encoder_proj.bias");

    embed_lin_w_         = get("flow.encoder.embed.out.0.weight");
    embed_lin_b_         = get("flow.encoder.embed.out.0.bias");
    embed_ln_w_          = get("flow.encoder.embed.out.1.weight");
    embed_ln_b_          = get("flow.encoder.embed.out.1.bias");

    up_embed_lin_w_      = get("flow.encoder.up_embed.out.0.weight");
    up_embed_lin_b_      = get("flow.encoder.up_embed.out.0.bias");
    up_embed_ln_w_       = get("flow.encoder.up_embed.out.1.weight");
    up_embed_ln_b_       = get("flow.encoder.up_embed.out.1.bias");

    prelook_conv1_w_     = get("flow.encoder.pre_lookahead_layer.conv1.weight");
    prelook_conv1_b_     = get("flow.encoder.pre_lookahead_layer.conv1.bias");
    prelook_conv2_w_     = get("flow.encoder.pre_lookahead_layer.conv2.weight");
    prelook_conv2_b_     = get("flow.encoder.pre_lookahead_layer.conv2.bias");

    up_layer_conv_w_     = get("flow.encoder.up_layer.conv.weight");
    up_layer_conv_b_     = get("flow.encoder.up_layer.conv.bias");

    after_norm_w_        = get("flow.encoder.after_norm.weight");
    after_norm_b_        = get("flow.encoder.after_norm.bias");

    auto fetch_block = [&](Block& B, const char* prefix_fmt, int i) {
        char nm[160];
        auto f = [&](const char* suf) {
            std::snprintf(nm, sizeof(nm), prefix_fmt, i);
            std::string full = std::string(nm) + suf;
            return get(full.c_str());
        };
        B.norm_mha_w  = f(".norm_mha.weight");
        B.norm_mha_b  = f(".norm_mha.bias");
        B.norm_ff_w   = f(".norm_ff.weight");
        B.norm_ff_b   = f(".norm_ff.bias");
        B.lin_q_w     = f(".self_attn.linear_q.weight");
        B.lin_q_b     = f(".self_attn.linear_q.bias");
        B.lin_k_w     = f(".self_attn.linear_k.weight");
        B.lin_k_b     = f(".self_attn.linear_k.bias");
        B.lin_v_w     = f(".self_attn.linear_v.weight");
        B.lin_v_b     = f(".self_attn.linear_v.bias");
        B.lin_out_w   = f(".self_attn.linear_out.weight");
        B.lin_out_b   = f(".self_attn.linear_out.bias");
        B.lin_pos_w   = f(".self_attn.linear_pos.weight");
        B.pos_bias_u  = f(".self_attn.pos_bias_u");
        B.pos_bias_v  = f(".self_attn.pos_bias_v");
        B.ff_w1_w     = f(".feed_forward.w_1.weight");
        B.ff_w1_b     = f(".feed_forward.w_1.bias");
        B.ff_w2_w     = f(".feed_forward.w_2.weight");
        B.ff_w2_b     = f(".feed_forward.w_2.bias");
        return B.norm_mha_w && B.lin_q_w && B.ff_w1_w && B.pos_bias_u;
    };

    encoders_.resize(cfg_.n_blocks);
    for (int i = 0; i < cfg_.n_blocks; ++i) {
        if (!fetch_block(encoders_[i],
                          "flow.encoder.encoders.%d", i)) return false;
    }
    up_encoders_.resize(cfg_.n_up_blocks);
    for (int i = 0; i < cfg_.n_up_blocks; ++i) {
        if (!fetch_block(up_encoders_[i],
                          "flow.encoder.up_encoders.%d", i)) return false;
    }

    // Sanity: vocab + dims from the embedding table.
    if (input_embedding_w_->ne[0] != cfg_.d_model
        || input_embedding_w_->ne[1] != cfg_.vocab_size) {
        std::fprintf(stderr,
                     "FlowEncoder::load: input_embedding shape mismatch "
                     "ne=(%lld, %lld) vs expected (%d, %d)\n",
                     (long long)input_embedding_w_->ne[0],
                     (long long)input_embedding_w_->ne[1],
                     cfg_.d_model, cfg_.vocab_size);
        return false;
    }
    return true;
}

std::unique_ptr<FlowEncoder> FlowEncoder::load(Model* model) {
    if (!model) return nullptr;
    if (model->arch != "chatterbox_s3gen") {
        std::fprintf(stderr,
                     "FlowEncoder::load: expected chatterbox_s3gen arch, got %s\n",
                     model->arch.c_str());
        return nullptr;
    }
    auto fe = std::unique_ptr<FlowEncoder>(new FlowEncoder());
    fe->model_ = model;
    if (!fe->fetch_(model)) return nullptr;

    fe->backend_ = chatterbox::default_backend();
    if (!fe->backend_) {
        std::fprintf(stderr, "FlowEncoder::load: backend init failed\n");
        return nullptr;
    }
    std::printf("FlowEncoder::load: ok (%d encoder + %d up_encoder blocks)\n",
                fe->cfg_.n_blocks, fe->cfg_.n_up_blocks);
    return fe;
}

namespace {

// Conformer block forward. Input/output: (D, T) ne[0]=D ne[1]=T.
// `pos_emb_DL` has ne[0]=D, ne[1]=L=2T-1.
ggml_tensor* conformer_block_forward(
        ggml_context* ctx, ggml_tensor* x, ggml_tensor* pos_emb_DL,
        const FlowEncoder::Block& B, const FlowEncoderConfig& cfg,
        int T) {

    const int D  = cfg.d_model;
    const int H  = cfg.n_heads;
    const int Hd = cfg.d_head;

    // -------- MHA branch --------
    ggml_tensor* h = layer_norm(ctx, x, B.norm_mha_w, B.norm_mha_b,
                                  cfg.ln_eps_attn);

    // Q, K, V projections. Weights stored as (in=D, out=D) ggml ne.
    ggml_tensor* q = ggml_add(ctx, mm_f32(ctx, B.lin_q_w, h), B.lin_q_b);
    ggml_tensor* k = ggml_add(ctx, mm_f32(ctx, B.lin_k_w, h), B.lin_k_b);
    ggml_tensor* v = ggml_add(ctx, mm_f32(ctx, B.lin_v_w, h), B.lin_v_b);
    // Each: ne=(D, T).

    // Reshape to per-head: (Hd, H, T). ne[0]=Hd, ne[1]=H, ne[2]=T.
    ggml_tensor* qh = ggml_reshape_3d(ctx, q, Hd, H, T);
    ggml_tensor* kh = ggml_reshape_3d(ctx, k, Hd, H, T);
    ggml_tensor* vh = ggml_reshape_3d(ctx, v, Hd, H, T);

    // Add pos_bias_u and pos_bias_v to q.
    // pos_bias_{u,v} ggml ne=(Hd, H). Broadcast across T (ne[2]).
    ggml_tensor* pbu = B.pos_bias_u;
    ggml_tensor* pbv = B.pos_bias_v;
    if (pbu->type != GGML_TYPE_F32) {
        pbu = ggml_cast(ctx, pbu, GGML_TYPE_F32);
    }
    if (pbv->type != GGML_TYPE_F32) {
        pbv = ggml_cast(ctx, pbv, GGML_TYPE_F32);
    }
    // Make it broadcastable to (Hd, H, T) by viewing as (Hd, H, 1).
    ggml_tensor* pbu_3 = ggml_reshape_3d(ctx, pbu, Hd, H, 1);
    ggml_tensor* pbv_3 = ggml_reshape_3d(ctx, pbv, Hd, H, 1);

    ggml_tensor* q_u = ggml_add(ctx, qh, pbu_3);   // (Hd, H, T)
    ggml_tensor* q_v = ggml_add(ctx, qh, pbv_3);

    // Positional projection: pos_emb_DL is (D, L). linear_pos has weight
    // (in=D, out=D), no bias. Result is (D, L) again.
    ggml_tensor* p = mm_f32(ctx, B.lin_pos_w, pos_emb_DL);            // (D, L)
    const int L = static_cast<int>(p->ne[1]);                        // 2T-1
    ggml_tensor* ph = ggml_reshape_3d(ctx, p, Hd, H, L);              // (Hd, H, L)

    // Permute q_u, q_v from (Hd, H, T) to (Hd, T, H) for ggml_mul_mat with k.
    ggml_tensor* Qu_p = ggml_cont(ctx, ggml_permute(ctx, q_u, 0, 2, 1, 3));
    ggml_tensor* Qv_p = ggml_cont(ctx, ggml_permute(ctx, q_v, 0, 2, 1, 3));
    ggml_tensor* Kp   = ggml_cont(ctx, ggml_permute(ctx, kh,  0, 2, 1, 3));
    ggml_tensor* Pp   = ggml_cont(ctx, ggml_permute(ctx, ph,  0, 2, 1, 3));
    // Qu_p, Qv_p: (Hd, T, H). Kp: (Hd, T, H). Pp: (Hd, L, H).

    // matrix_ac = Q_u @ K^T  → (T, T, H)
    ggml_tensor* AC = mm_f32(ctx, Kp, Qu_p);            // (T_a, T_b, H)
    // matrix_bd = Q_v @ P^T  → (L, T, H)
    ggml_tensor* BD = mm_f32(ctx, Pp, Qv_p);            // (L, T, H)

    // Apply rel_shift to BD (skip if shapes already match — only when
    // L == T, which doesn't happen in our setup).
    ggml_tensor* BD_shift = rel_shift(ctx, BD, T, H);   // (T, T, H)

    // Sum, scale, softmax.
    ggml_tensor* scores = ggml_add(ctx, AC, BD_shift);   // (T, T, H)
    scores = ggml_scale(ctx, scores, 1.0f / std::sqrt(static_cast<float>(Hd)));
    scores = ggml_soft_max(ctx, scores);

    // Apply to V. V layout for the mul: (T, Hd, H). Permute vh
    // (Hd, H, T) → (T, Hd, H) via (1, 2, 0, 3): no, let me think.
    // vh has ne=(Hd, H, T). We want a (T, Hd, H) tensor for the matmul.
    // Permutation axes (0,1,2) → (new0=T, new1=Hd, new2=H).
    // Mapping: old (0=Hd, 1=H, 2=T) -> new (0=T, 1=Hd, 2=H). So perm = (2, 0, 1).
    ggml_tensor* V_trans = ggml_cont_3d(ctx,
        ggml_permute(ctx, vh, 1, 2, 0, 3), T, Hd, H);

    // ggml_mul_mat(V_trans, scores): a=V_trans (T, Hd, H) → rows = Hd?
    // Wait: ggml_mul_mat(a, b) computes a^T @ b along ne[0]. a is (k=T, Hd, H),
    // b is (T, T, H). Result: (Hd, T, H). That matches numpy's attn @ v.
    ggml_tensor* attn = mm_f32(ctx, V_trans, scores);         // (Hd, T, H)
    // Permute back to (Hd, H, T) — perm (0, 2, 1, 3).
    ggml_tensor* attn_p = ggml_cont(ctx, ggml_permute(ctx, attn, 0, 2, 1, 3));
    // Flatten heads: (D, T).
    ggml_tensor* attn_flat = ggml_reshape_2d(ctx, attn_p, D, T);

    // Linear out.
    ggml_tensor* attn_out = ggml_add(ctx,
        mm_f32(ctx, B.lin_out_w, attn_flat), B.lin_out_b);

    // Residual.
    x = ggml_add(ctx, x, attn_out);

    // -------- FF branch --------
    ggml_tensor* h2 = layer_norm(ctx, x, B.norm_ff_w, B.norm_ff_b,
                                   cfg.ln_eps_attn);
    h2 = ggml_add(ctx, mm_f32(ctx, B.ff_w1_w, h2), B.ff_w1_b);
    h2 = ggml_silu(ctx, h2);                              // Swish = SiLU
    h2 = ggml_add(ctx, mm_f32(ctx, B.ff_w2_w, h2), B.ff_w2_b);
    return ggml_add(ctx, x, h2);
}

// LinearNoSubsampling block: Linear (D -> D) + LayerNorm (eps=1e-5) +
// xscale (sqrt(D)). Input/output: (D, T).
ggml_tensor* linear_no_subsample(
        ggml_context* ctx, ggml_tensor* x,
        ggml_tensor* lin_w, ggml_tensor* lin_b,
        ggml_tensor* ln_w,  ggml_tensor* ln_b,
        const FlowEncoderConfig& cfg) {
    x = ggml_add(ctx, mm_f32(ctx, lin_w, x), lin_b);
    x = layer_norm(ctx, x, ln_w, ln_b, cfg.ln_eps);
    x = ggml_scale(ctx, x, std::sqrt(static_cast<float>(cfg.d_model)));
    return x;
}

// pre_lookahead_layer forward. Input/output: (D, T).
//
// Internally works in (T, D) layout (conv1d's natural form), so we
// transpose to (T, D) at the boundary and back at the end.
//
//   x_TD = transpose(x)                          # (T, D)
//   x_TD = right-pad by pre_lookahead_len        # (T + pre_lookahead, D)
//   y = conv1d(x_TD, conv1_w, k=4, no-pad)       # (T, D) since kernel=4, pad=3 right=>(T+3-4+1=T)
//                                                  Wait: input length T+pre_lookahead = T+3.
//                                                  After conv k=4 no pad: T+3-4+1 = T. ✓
//   y = leaky_relu(y + conv1_b)
//   y = left-pad by 2                            # (T+2, D)
//   y = conv1d(y, conv2_w, k=3, no-pad)          # (T+2-3+1 = T, D). ✓
//   y = y + conv2_b
//   y_DT = transpose(y) + x                       # residual on the input
ggml_tensor* pre_lookahead_forward(
        ggml_context* ctx, ggml_tensor* x,
        ggml_tensor* c1_w, ggml_tensor* c1_b,
        ggml_tensor* c2_w, ggml_tensor* c2_b,
        const FlowEncoderConfig& cfg, int T) {
    const int D    = cfg.d_model;
    const int pkn  = cfg.pre_lookahead_len;

    // To (T, D).
    ggml_tensor* x_TD = ggml_cont(ctx, ggml_transpose(ctx, x));

    // Right-pad by pkn=3. ggml_pad pads on the high side of each dim.
    // We want to extend ne[0] (T axis) by pkn. ggml_pad signature:
    //   ggml_pad(ctx, a, p0, p1, p2, p3) pads ne[0] by p0, ne[1] by p1, etc.
    ggml_tensor* x_pad1 = ggml_pad(ctx, x_TD, pkn, 0, 0, 0);     // (T+pkn, D)

    // conv1 (kernel=4, stride=1, no pad).
    ggml_tensor* y = conv_1d_f32(ctx, c1_w, x_pad1, 1, 0, 1);   // (T, D)
    y = ggml_add(ctx, y, bias_for_TC(ctx, c1_b, D));
    y = ggml_leaky_relu(ctx, y, 0.01f, false);

    // Left-pad by 2 along the T axis. ggml_pad pads only right; emulate
    // via concat with a zero (2, D) at the front.
    //
    // Strategy: take a (2, D) view of y (offset 0), scale by 0, concat
    // along ne[0] = the T axis with y.
    ggml_tensor* zero_2D_view = ggml_view_2d(ctx, y, 2, D,
                                                /*nb1=*/y->nb[1],
                                                /*offset=*/0);
    ggml_tensor* zero_2D = ggml_scale(ctx,
                            ggml_cont(ctx, zero_2D_view), 0.0f);   // (2, D)
    ggml_tensor* y_pad2 = ggml_concat(ctx, zero_2D, y, /*dim=*/0);  // (T+2, D)

    // conv2 (kernel=3, stride=1, no pad).
    ggml_tensor* y2 = conv_1d_f32(ctx, c2_w, y_pad2, 1, 0, 1);    // (T, D)
    y2 = ggml_add(ctx, y2, bias_for_TC(ctx, c2_b, D));

    // Back to (D, T) and residual.
    ggml_tensor* y_DT = ggml_cont(ctx, ggml_transpose(ctx, y2));   // (D, T)
    return ggml_add(ctx, y_DT, x);
}

// up_layer forward. Input (D, T) → output (D, 2T).
//
//   x_TD = transpose(x)                          # (T, D)
//   # nearest 2x interpolate over T: each row repeated `stride` times
//   x_TD = repeat-interleave on ne[0]            # (2T, D)
//   # left-pad by stride*2 = 4
//   x_TD = left-pad-4                            # (2T+4, D)
//   y    = conv1d(x_TD, up_conv_w, k=5, no-pad)  # (2T, D)
//   y    = y + up_conv_b
//   y_DT = transpose(y)                           # (D, 2T)
ggml_tensor* up_layer_forward(ggml_context* ctx, ggml_tensor* x,
                                ggml_tensor* uc_w, ggml_tensor* uc_b,
                                const FlowEncoderConfig& cfg, int T) {
    const int D = cfg.d_model;
    const int S = cfg.up_stride;     // 2

    // To (T, D).
    ggml_tensor* x_TD = ggml_cont(ctx, ggml_transpose(ctx, x));

    // Nearest 2x interpolate: repeat each row S=2 times along ne[0].
    // ggml_upscale_ext upscales by integer factor on selected dims.
    // Signature: ggml_upscale_ext(ctx, a, ne0, ne1, ne2, ne3, mode).
    // Hmm, prefer ggml_repeat which expects shape arg. But simpler:
    // use ggml_upscale (or its alias) that does nearest-neighbor on
    // ne[0] / ne[1]. Fall back to manual: tile in ne[0] by S.
    //
    // The cleanest portable approach: reshape (T, D) -> (1, T, D, 1),
    // upscale ne[1] by S, reshape back. ggml_upscale_ext is the right
    // tool — it broadcasts each element of ne[0..3] to the new sizes
    // with nearest-neighbor interpolation.
    ggml_tensor* up = ggml_interpolate(ctx, x_TD,
                                         /*ne0=*/S * T,
                                         /*ne1=*/D,
                                         /*ne2=*/1, /*ne3=*/1,
                                         GGML_SCALE_MODE_NEAREST);
    // up: (2T, D)

    // Left-pad by S*2 = 4 along ne[0]. Same trick as pre_lookahead.
    ggml_tensor* zero_view = ggml_view_2d(ctx, up,
                                             S * 2, D,
                                             /*nb1=*/up->nb[1],
                                             /*offset=*/0);
    ggml_tensor* zero_pad = ggml_scale(ctx, ggml_cont(ctx, zero_view), 0.0f);
    ggml_tensor* padded   = ggml_concat(ctx, zero_pad, up, /*dim=*/0);
    // padded: (2T + 4, D)

    // conv1d kernel=5, stride=1, no pad.
    ggml_tensor* y = conv_1d_f32(ctx, uc_w, padded, 1, 0, 1);   // (2T, D)
    y = ggml_add(ctx, y, bias_for_TC(ctx, uc_b, D));

    return ggml_cont(ctx, ggml_transpose(ctx, y));   // (D, 2T)
}

}  // namespace


// ----------------------------------------------------------------------------
// Public forward (full pipeline)
// ----------------------------------------------------------------------------

std::vector<float> FlowEncoder::forward_with_stages(
        const std::vector<int32_t>& tokens,
        std::unordered_map<std::string, std::vector<float>>& stages_out,
        std::unordered_map<std::string, std::pair<int, int>>& stage_shapes) {
    stages_out.clear();
    stage_shapes.clear();
    if (tokens.empty()) return {};

    const int T   = static_cast<int>(tokens.size());
    const int D   = cfg_.d_model;
    const int T2  = 2 * T;
    const int L1  = 2 * T - 1;
    const int L2  = 2 * T2 - 1;

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

    // ---- Inputs ----
    ggml_tensor* tokens_in = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
    ggml_set_input(tokens_in);
    ggml_set_name(tokens_in, "tokens");

    ggml_tensor* pos1_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, L1);
    ggml_set_input(pos1_in);
    ggml_set_name(pos1_in, "pos1");

    ggml_tensor* pos2_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, L2);
    ggml_set_input(pos2_in);
    ggml_set_name(pos2_in, "pos2");

    // Convenience for marking stages — register the named output, save the
    // ggml_tensor* for later get_back, then continue.
    std::vector<std::pair<std::string, ggml_tensor*>> stage_tensors;
    auto mark = [&](const char* name, ggml_tensor* t) {
        ggml_set_name(t, name);
        ggml_set_output(t);
        stage_tensors.emplace_back(name, t);
    };

    // ---- input_embedding ----
    // input_embedding_w_ has ne=(D, vocab). For ggml_get_rows, rows are on
    // ne[1]; result ne=(D, T). Matches our (D, T) layout.
    ggml_tensor* x = ggml_get_rows(ctx, input_embedding_w_, tokens_in);
    // Cast to fp32 if the embedding table is fp16.
    if (x->type != GGML_TYPE_F32) x = ggml_cast(ctx, x, GGML_TYPE_F32);
    mark("after_input_embedding", x);

    // ---- encoder.embed (Linear + LN + xscale) ----
    x = linear_no_subsample(ctx, x,
                              embed_lin_w_, embed_lin_b_,
                              embed_ln_w_,  embed_ln_b_,
                              cfg_);
    mark("after_embed", x);

    // ---- pre_lookahead ----
    x = pre_lookahead_forward(ctx, x,
                                prelook_conv1_w_, prelook_conv1_b_,
                                prelook_conv2_w_, prelook_conv2_b_,
                                cfg_, T);
    mark("after_prelookahead", x);

    // ---- 6× conformer blocks @ 25 Hz ----
    for (int i = 0; i < cfg_.n_blocks; ++i) {
        x = conformer_block_forward(ctx, x, pos1_in, encoders_[i], cfg_, T);
        if (i == 0)                    mark("after_enc_block0", x);
        if (i == cfg_.n_blocks - 1)    mark("after_enc_block5", x);
    }

    // ---- up_layer ----
    x = up_layer_forward(ctx, x,
                           up_layer_conv_w_, up_layer_conv_b_,
                           cfg_, T);
    mark("after_uplayer", x);

    // ---- up_embed ----
    x = linear_no_subsample(ctx, x,
                              up_embed_lin_w_, up_embed_lin_b_,
                              up_embed_ln_w_,  up_embed_ln_b_,
                              cfg_);
    mark("after_upembed", x);

    // ---- 4× conformer blocks @ 50 Hz ----
    for (int i = 0; i < cfg_.n_up_blocks; ++i) {
        x = conformer_block_forward(ctx, x, pos2_in, up_encoders_[i],
                                       cfg_, T2);
        if (i == 0)                       mark("after_upenc_block0", x);
        if (i == cfg_.n_up_blocks - 1)    mark("after_upenc_block3", x);
    }

    // ---- after_norm ----
    x = layer_norm(ctx, x, after_norm_w_, after_norm_b_, cfg_.ln_eps);
    mark("after_afternorm", x);

    // ---- encoder_proj (Linear 512 -> 80) ----
    x = ggml_add(ctx, mm_f32(ctx, encoder_proj_w_, x),
                 encoder_proj_b_);                                  // (80, 2T)
    mark("after_encoderproj", x);

    ggml_build_forward_expand(gf, x);

    // ---- Allocate + fill inputs ----
    ggml_backend_sched_t sched = chatterbox::make_sched(GRAPH_MAX_NODES);
    // Parity is held by per-matmul GGML_PREC_F32 in mm_f32 / conv_1d_f32
    // (used throughout conformer_block_forward, pre_lookahead_forward,
    // up_layer_forward, and encoder_proj). No CPU pin needed.
    if (!sched || !ggml_backend_sched_alloc_graph(sched, gf)) {
        std::fprintf(stderr,
                     "FlowEncoder::forward_with_stages: galloc_alloc_graph failed\n");
        if (sched) ggml_backend_sched_free(sched);
        ggml_free(ctx);
        return {};
    }

    ggml_backend_tensor_set(tokens_in, tokens.data(), 0,
                             T * sizeof(int32_t));

    const auto pos1 = build_rel_pos_slice(T,  D, MAX_REL_LEN);
    const auto pos2 = build_rel_pos_slice(T2, D, MAX_REL_LEN);
    ggml_backend_tensor_set(pos1_in, pos1.data(), 0,
                             pos1.size() * sizeof(float));
    ggml_backend_tensor_set(pos2_in, pos2.data(), 0,
                             pos2.size() * sizeof(float));

    if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr,
                     "FlowEncoder::forward_with_stages: graph_compute failed\n");
        if (sched) ggml_backend_sched_free(sched);
        ggml_free(ctx);
        return {};
    }

    // ---- Read every stage ----
    for (const auto& [name, t] : stage_tensors) {
        // Stage tensor shapes: ggml (C, T) ne[0]=C, ne[1]=T (transformer
        // layout). Convert to numpy (T, C) by reading raw bytes — since
        // ggml ne[0] is innermost = numpy last dim, byte memory is the
        // same row-major (T, C).
        const int C_st = static_cast<int>(t->ne[0]);
        const int T_st = static_cast<int>(t->ne[1]);
        std::vector<float> out_st(static_cast<size_t>(C_st) * T_st);
        ggml_backend_tensor_get(t, out_st.data(), 0,
                                  out_st.size() * sizeof(float));
        stages_out[name] = std::move(out_st);
        stage_shapes[name] = {T_st, C_st};
    }

    // Final output = after_encoderproj.
    const auto& final_buf  = stages_out.at("after_encoderproj");
    auto final_buf_copy = final_buf;
    if (sched) ggml_backend_sched_free(sched);
    ggml_free(ctx);
    return final_buf_copy;
}

std::vector<float> FlowEncoder::forward(const std::vector<int32_t>& tokens,
                                          int& out_T_out, int& out_d) {
    std::unordered_map<std::string, std::vector<float>> stages;
    std::unordered_map<std::string, std::pair<int, int>> shapes;
    auto out = forward_with_stages(tokens, stages, shapes);
    if (auto it = shapes.find("after_encoderproj"); it != shapes.end()) {
        out_T_out = it->second.first;
        out_d     = it->second.second;
    } else {
        out_T_out = 0; out_d = 0;
    }
    return out;
}

std::vector<float> FlowEncoder::affine_speaker(const std::vector<float>& spk_192) {
    if (static_cast<int>(spk_192.size()) != cfg_.spk_in) return {};

    const size_t buf_size =
        ggml_tensor_overhead() * 32
        + ggml_graph_overhead_custom(32, false);
    std::vector<uint8_t> buf(buf_size);
    ggml_init_params params{};
    params.mem_size = buf.size();
    params.mem_buffer = buf.data();
    params.no_alloc = true;
    ggml_context* ctx = ggml_init(params);
    if (!ctx) return {};
    ggml_cgraph* gf = ggml_new_graph_custom(ctx, 32, false);

    // Input: (192, 1)
    ggml_tensor* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cfg_.spk_in, 1);
    ggml_set_input(x);

    // spk_affine_w_ has ne=(192, 80). y = W @ x → (80, 1).
    ggml_tensor* y = ggml_add(ctx, ggml_mul_mat(ctx, spk_affine_w_, x),
                                 spk_affine_b_);
    ggml_set_output(y);
    ggml_build_forward_expand(gf, y);

    ggml_backend_sched_t sched = chatterbox::make_sched(GRAPH_MAX_NODES);
    if (!sched || !ggml_backend_sched_alloc_graph(sched, gf)) {
        if (sched) ggml_backend_sched_free(sched);
        ggml_free(ctx);
        return {};
    }
    ggml_backend_tensor_set(x, spk_192.data(), 0,
                             spk_192.size() * sizeof(float));
    if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) {
        if (sched) ggml_backend_sched_free(sched);
        ggml_free(ctx);
        return {};
    }
    std::vector<float> out(cfg_.d_out);
    ggml_backend_tensor_get(y, out.data(), 0, out.size() * sizeof(float));
    if (sched) ggml_backend_sched_free(sched);
    ggml_free(ctx);
    return out;
}

}  // namespace chatterbox
