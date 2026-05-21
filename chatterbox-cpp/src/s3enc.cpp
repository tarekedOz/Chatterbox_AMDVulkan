#include "s3enc.h"
#include "backend.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <cmath>
#include <cstdio>
#include <cstring>

namespace chatterbox {

namespace {

constexpr size_t GRAPH_MAX_NODES = 32768;

uint32_t read_u32(gguf_context* gguf, const char* key, uint32_t def) {
    int64_t id = gguf_find_key(gguf, key);
    return id < 0 ? def : gguf_get_val_u32(gguf, id);
}

}  // namespace

S3Encoder::~S3Encoder() {
    // backend_ is a process-wide singleton (see backend.h); do NOT free.
}

std::unique_ptr<S3Encoder> S3Encoder::load(Model* model) {
    if (!model) return nullptr;
    if (model->arch != "chatterbox_s3gen") {
        std::fprintf(stderr,
                     "S3Encoder::load: expected chatterbox_s3gen arch, got %s\n",
                     model->arch.c_str());
        return nullptr;
    }
    auto enc = std::unique_ptr<S3Encoder>(new S3Encoder());
    enc->model_ = model;

    auto get = [&](const char* name) -> ggml_tensor* {
        ggml_tensor* t = model->find_tensor(name);
        if (!t) std::fprintf(stderr, "S3Encoder::load: missing %s\n", name);
        return t;
    };

    enc->conv1_w_ = get("tokenizer.encoder.conv1.weight");
    enc->conv1_b_ = get("tokenizer.encoder.conv1.bias");
    enc->conv2_w_ = get("tokenizer.encoder.conv2.weight");
    enc->conv2_b_ = get("tokenizer.encoder.conv2.bias");
    if (!enc->conv1_w_ || !enc->conv1_b_ || !enc->conv2_w_ || !enc->conv2_b_) {
        return nullptr;
    }

    enc->blocks_.resize(enc->cfg_.n_layer);
    char nm[80];
    auto fetch_block = [&](Block& b, int i) {
        auto f = [&](const char* fmt) {
            std::snprintf(nm, sizeof(nm), fmt, i);
            return get(nm);
        };
        b.attn_ln_w   = f("tokenizer.encoder.blocks.%d.attn_ln.weight");
        b.attn_ln_b   = f("tokenizer.encoder.blocks.%d.attn_ln.bias");
        b.q_w         = f("tokenizer.encoder.blocks.%d.attn.query.weight");
        b.q_b         = f("tokenizer.encoder.blocks.%d.attn.query.bias");
        b.k_w         = f("tokenizer.encoder.blocks.%d.attn.key.weight");
        b.v_w         = f("tokenizer.encoder.blocks.%d.attn.value.weight");
        b.v_b         = f("tokenizer.encoder.blocks.%d.attn.value.bias");
        b.out_w       = f("tokenizer.encoder.blocks.%d.attn.out.weight");
        b.out_b       = f("tokenizer.encoder.blocks.%d.attn.out.bias");
        b.fsmn_w      = f("tokenizer.encoder.blocks.%d.attn.fsmn_block.weight");
        b.mlp_ln_w    = f("tokenizer.encoder.blocks.%d.mlp_ln.weight");
        b.mlp_ln_b    = f("tokenizer.encoder.blocks.%d.mlp_ln.bias");
        b.mlp_fc_w    = f("tokenizer.encoder.blocks.%d.mlp.0.weight");
        b.mlp_fc_b    = f("tokenizer.encoder.blocks.%d.mlp.0.bias");
        b.mlp_proj_w  = f("tokenizer.encoder.blocks.%d.mlp.2.weight");
        b.mlp_proj_b  = f("tokenizer.encoder.blocks.%d.mlp.2.bias");
        return b.attn_ln_w && b.attn_ln_b && b.q_w && b.q_b && b.k_w
            && b.v_w && b.v_b && b.out_w && b.out_b && b.fsmn_w
            && b.mlp_ln_w && b.mlp_ln_b && b.mlp_fc_w && b.mlp_fc_b
            && b.mlp_proj_w && b.mlp_proj_b;
    };
    for (int i = 0; i < enc->cfg_.n_layer; ++i) {
        if (!fetch_block(enc->blocks_[i], i)) return nullptr;
    }

    enc->backend_ = chatterbox::default_backend();
    if (!enc->backend_) {
        std::fprintf(stderr, "S3Encoder::load: ggml_backend_cpu_init failed\n");
        return nullptr;
    }
    return enc;
}

namespace {

// LayerNorm with gain + bias, applied along ne[0] (the feature axis when
// data is laid out as (D, T)).
ggml_tensor* layer_norm(ggml_context* ctx, ggml_tensor* x,
                         ggml_tensor* w, ggml_tensor* b, float eps) {
    ggml_tensor* n = ggml_norm(ctx, x, eps);
    return ggml_add(ctx, ggml_mul(ctx, n, w), b);
}

// Add a 1D bias (ne[0]=D) to a 2D tensor with ne[0]=D, ne[1]=T.
// ggml_add broadcasts the missing ne[1]=1 across the time axis.
// This is the natural layout for mul_mat outputs (features on ne[0]).
ggml_tensor* add_bias_dt(ggml_context* ctx, ggml_tensor* x, ggml_tensor* b) {
    return ggml_add(ctx, x, b);
}

// Add a 1D bias (ne[0]=C) to a conv1d output with ne[0]=T, ne[1]=C.
// Conv1d outputs use the OPPOSITE convention (time on ne[0], channels on
// ne[1]) so the bias has to be viewed as (1, C) before the add for ggml's
// broadcast rule to see it as ne[0]=1 (broadcast over T).
ggml_tensor* add_bias_td(ggml_context* ctx, ggml_tensor* x, ggml_tensor* b,
                          int n_channels) {
    return ggml_add(ctx, x, ggml_reshape_2d(ctx, b, 1, n_channels));
}

}  // namespace

std::vector<float> S3Encoder::forward(const std::vector<float>& log_mel,
                                       int n_mels, int T_mel,
                                       int& out_T_tok, int& out_n_state) {
    out_T_tok = 0;
    out_n_state = 0;
    if (n_mels != cfg_.n_mels) {
        std::fprintf(stderr,
                     "S3Encoder::forward: expected n_mels=%d, got %d\n",
                     cfg_.n_mels, n_mels);
        return {};
    }
    if (T_mel <= 0
        || static_cast<int>(log_mel.size()) != n_mels * T_mel) {
        std::fprintf(stderr,
                     "S3Encoder::forward: bad mel buffer (%zu floats vs %d*%d)\n",
                     log_mel.size(), n_mels, T_mel);
        return {};
    }

    const int D  = cfg_.n_state;
    const int H  = cfg_.n_head;
    const int Hd = cfg_.head_dim;
    const float eps = cfg_.layer_norm_eps;

    const size_t buf_size =
        ggml_tensor_overhead() * GRAPH_MAX_NODES
        + ggml_graph_overhead_custom(GRAPH_MAX_NODES, false);
    std::vector<uint8_t> buf(buf_size);

    ggml_init_params params{};
    params.mem_size   = buf.size();
    params.mem_buffer = buf.data();
    params.no_alloc   = true;

    ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        std::fprintf(stderr, "S3Encoder::forward: ggml_init failed\n");
        return {};
    }
    ggml_cgraph* gf = ggml_new_graph_custom(ctx, GRAPH_MAX_NODES, false);

    // --- Inputs ---
    // mel: ne[0]=T_mel (time), ne[1]=n_mels — matches ggml's conv1d input layout
    // where channels live on ne[1] and length on ne[0].
    ggml_tensor* mel = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T_mel, n_mels);
    ggml_set_name(mel, "mel");
    ggml_set_input(mel);

    // --- Conv1 + GELU(erf) ---
    // kernel shape (k=3, in=128, out=1280) -> ggml ne=[3, 128, 1280]. Matches.
    ggml_tensor* x = ggml_conv_1d(ctx, conv1_w_, mel, /*s=*/2, /*p=*/1, /*d=*/1);
    // x: ne[0]=T_mel/2, ne[1]=1280  (conv1d's time-on-ne[0] convention)
    x = add_bias_td(ctx, x, conv1_b_, D);
    x = ggml_gelu_erf(ctx, x);

    // --- Conv2 + GELU(erf) ---
    x = ggml_conv_1d(ctx, conv2_w_, x, /*s=*/2, /*p=*/1, /*d=*/1);
    x = add_bias_td(ctx, x, conv2_b_, D);
    x = ggml_gelu_erf(ctx, x);
    // x: ne[0]=T_tok, ne[1]=1280

    const int T_tok = static_cast<int>(x->ne[0]);

    // Transition to the transformer's (D, T_tok) layout — features on
    // ne[0] matches the mul_mat convention used throughout T3 / VE.
    x = ggml_cont(ctx, ggml_transpose(ctx, x));   // (D, T_tok)

    // Positions for RoPE (i32, length T_tok).
    ggml_tensor* positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T_tok);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);

    for (int il = 0; il < cfg_.n_layer; ++il) {
        const Block& B = blocks_[il];

        // --- Pre-attention LN ---
        ggml_tensor* x_n = layer_norm(ctx, x, B.attn_ln_w, B.attn_ln_b, eps);
        // x_n: (D, T_tok)

        // --- QKV projections (Linear; weights stored (out, in) -> ggml
        // ne[0]=in=D, ne[1]=out=D, so no transpose needed). ---
        ggml_tensor* q = ggml_add(ctx, ggml_mul_mat(ctx, B.q_w, x_n), B.q_b);
        ggml_tensor* k = ggml_mul_mat(ctx, B.k_w, x_n);  // no bias
        ggml_tensor* v = ggml_add(ctx, ggml_mul_mat(ctx, B.v_w, x_n), B.v_b);
        // Each: (D, T_tok)

        // Reshape to per-head: (Hd, H, T_tok). RoPE wants ne[0]=head_dim
        // contiguous (the rotation axis), then heads, then tokens.
        ggml_tensor* qh = ggml_reshape_3d(ctx, q, Hd, H, T_tok);
        ggml_tensor* kh = ggml_reshape_3d(ctx, k, Hd, H, T_tok);

        // Apply RoPE (NEOX half-split, freq_base=10000) to Q and K.
        const int n_dims = Hd;
        qh = ggml_rope_ext(ctx, qh, positions, nullptr,
                            n_dims, GGML_ROPE_TYPE_NEOX,
                            /*n_ctx_orig=*/cfg_.rope_max,
                            /*freq_base=*/cfg_.rope_base,
                            /*freq_scale=*/1.0f,
                            /*ext_factor=*/0.0f,
                            /*attn_factor=*/1.0f,
                            /*beta_fast=*/0.0f,
                            /*beta_slow=*/0.0f);
        kh = ggml_rope_ext(ctx, kh, positions, nullptr,
                            n_dims, GGML_ROPE_TYPE_NEOX,
                            cfg_.rope_max, cfg_.rope_base,
                            1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        // --- FSMN memory branch on V ---
        // Depthwise Conv1d(D, D, kernel=31, groups=D) plus a V residual,
        // per upstream's forward_fsmn. ggml_conv_1d_dw uses conv1d's
        // (T, C) layout, so transpose V (currently (D, T_tok) from mul_mat)
        // into (T_tok, D), apply, and transpose back to match `v` for the
        // residual add.
        ggml_tensor* v_td = ggml_cont(ctx, ggml_transpose(ctx, v));  // (T_tok, D)
        ggml_tensor* fsmn_out = ggml_conv_1d_dw(ctx, B.fsmn_w, v_td,
                                                  /*s=*/1, /*p=*/(cfg_.fsmn_k - 1) / 2,
                                                  /*d=*/1);
        // fsmn_out: (T_tok, D). Bring back to (D, T_tok) for the +V residual.
        fsmn_out = ggml_cont(ctx, ggml_transpose(ctx, fsmn_out));
        fsmn_out = ggml_add(ctx, fsmn_out, v);

        // --- Attention scores ---
        // Permute Q, K from (Hd, H, T) to (Hd, T, H) for the matmul pattern.
        ggml_tensor* Qp = ggml_permute(ctx, qh, 0, 2, 1, 3);   // (Hd, T, H)
        ggml_tensor* Kp = ggml_permute(ctx, kh, 0, 2, 1, 3);

        // V_trans for the final attn @ V mul_mat: (T, Hd, H) layout.
        ggml_tensor* vh = ggml_reshape_3d(ctx, v, Hd, H, T_tok);
        ggml_tensor* V_trans = ggml_cont_3d(ctx,
            ggml_permute(ctx, vh, 1, 2, 0, 3),   // (H, T, Hd) view
            T_tok, Hd, H);                        // contig (T, Hd, H)

        // K * Q -> (T, T, H). Scale by head_dim^-0.5 (upstream applies
        // head_dim^-0.25 on each of Q and K; ggml_mul_mat after that gives
        // the same effective scaling).
        ggml_tensor* KQ = ggml_mul_mat(ctx, Kp, Qp);
        KQ = ggml_scale(ctx, KQ, 1.0f / std::sqrt(static_cast<float>(Hd)));
        // No padding mask (single utterance, no padding tokens).
        KQ = ggml_soft_max(ctx, KQ);

        ggml_tensor* KQV = ggml_mul_mat(ctx, V_trans, KQ);     // (Hd, T, H)
        ggml_tensor* KQV_merged = ggml_permute(ctx, KQV, 0, 2, 1, 3);  // (Hd, H, T)
        ggml_tensor* attn = ggml_cont_2d(ctx, KQV_merged, D, T_tok);

        // --- Output projection + FSMN add ---
        ggml_tensor* attn_out = ggml_add(ctx, ggml_mul_mat(ctx, B.out_w, attn), B.out_b);
        attn_out = ggml_add(ctx, attn_out, fsmn_out);

        // Residual.
        x = ggml_add(ctx, x, attn_out);

        // --- Pre-MLP LN ---
        ggml_tensor* x_n2 = layer_norm(ctx, x, B.mlp_ln_w, B.mlp_ln_b, eps);

        // --- MLP: Linear -> GELU(erf) -> Linear ---
        ggml_tensor* h = ggml_add(ctx, ggml_mul_mat(ctx, B.mlp_fc_w, x_n2), B.mlp_fc_b);
        h = ggml_gelu_erf(ctx, h);
        h = ggml_add(ctx, ggml_mul_mat(ctx, B.mlp_proj_w, h), B.mlp_proj_b);

        x = ggml_add(ctx, x, h);
    }

    // Output: x is (D, T_tok) with ne[0]=D contiguous. The raw byte
    // layout matches numpy (T_tok, D) row-major exactly — ggml's
    // ne[0] is the innermost stride which is numpy's last (column) axis.
    // No extra transpose needed at the boundary.
    ggml_set_name(x, "hidden");
    ggml_set_output(x);
    ggml_build_forward_expand(gf, x);

    // Pin the encoder output to CPU. The 6-block attention stack
    // accumulates ~1e-1 of fp16 drift on Vulkan, enough to flip
    // discrete FSQ codes at the host-side round() in S3Tokenizer.
    // sched propagates the CPU placement backward as far as it can,
    // and copies Vulkan-resident weights to CPU as needed.
    ggml_backend_sched_t sched = chatterbox::make_sched(GRAPH_MAX_NODES);
    if (sched && chatterbox::is_vulkan_active()) {
        ggml_backend_sched_set_tensor_backend(
            sched, x, chatterbox::cpu_backend());
    }
    if (!sched || !ggml_backend_sched_alloc_graph(sched, gf)) {
        std::fprintf(stderr, "S3Encoder::forward: gallocr_alloc_graph failed\n");
        if (sched) ggml_backend_sched_free(sched);
        ggml_free(ctx);
        return {};
    }

    // Fill inputs.
    // NumPy reference layout: log_mel is (n_mels, T_mel) row-major
    //   = element [m, t] at offset m * T_mel + t.
    // ggml mel layout: ne[0]=T_mel, ne[1]=n_mels — contiguous fp32 with
    //   element [t, m] at offset (m * T_mel + t) * 4 bytes. Same memory order!
    // So we can memcpy log_mel directly into the mel tensor.
    ggml_backend_tensor_set(mel, log_mel.data(), 0,
                            log_mel.size() * sizeof(float));

    std::vector<int32_t> pos_ids(T_tok);
    for (int t = 0; t < T_tok; ++t) pos_ids[t] = t;
    ggml_backend_tensor_set(positions, pos_ids.data(), 0,
                            T_tok * sizeof(int32_t));

    if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "S3Encoder::forward: backend_graph_compute failed\n");
        if (sched) ggml_backend_sched_free(sched);
        ggml_free(ctx);
        return {};
    }

    out_T_tok   = T_tok;
    out_n_state = D;
    std::vector<float> result(static_cast<size_t>(T_tok) * D);
    ggml_backend_tensor_get(x, result.data(), 0, result.size() * sizeof(float));

    if (sched) ggml_backend_sched_free(sched);
    ggml_free(ctx);
    return result;
}

}  // namespace chatterbox
