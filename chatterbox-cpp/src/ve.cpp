#include "ve.h"
#include "backend.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace chatterbox {

namespace {

// LSTM unrolls one cell per frame * 3 layers. ~30 ops/cell, so e.g.
// 5 seconds of 16 kHz audio = 500 frames -> ~45K ops. Caller is
// expected to truncate to <= ~10 sec before this point.
constexpr size_t GRAPH_MAX_NODES = 262144;

uint32_t read_u32(gguf_context* gguf, const char* key, uint32_t fallback) {
    int64_t id = gguf_find_key(gguf, key);
    return id < 0 ? fallback : gguf_get_val_u32(gguf, id);
}

}  // namespace

VE::~VE() {
    // backend_ is a process-wide singleton (see backend.h); do NOT free.
}

std::unique_ptr<VE> VE::load(Model* model) {
    if (!model) return nullptr;
    if (model->arch != "chatterbox_ve") {
        std::fprintf(stderr,
                     "VE::load: expected arch chatterbox_ve, got %s\n",
                     model->arch.c_str());
        return nullptr;
    }

    auto ve = std::unique_ptr<VE>(new VE());
    ve->model_ = model;

    gguf_context* gguf = model->gguf();
    ve->cfg_.n_mels        = (int32_t)read_u32(gguf, "chatterbox_ve.n_mels",         40);
    ve->cfg_.hidden        = (int32_t)read_u32(gguf, "chatterbox_ve.hidden_size",    256);
    ve->cfg_.n_lstm_layers = (int32_t)read_u32(gguf, "chatterbox_ve.lstm_layers",    3);
    ve->cfg_.emb_dim       = (int32_t)read_u32(gguf, "chatterbox_ve.speaker_emb_dim",256);
    // final_relu: read via the bool key if present, else assume true.
    {
        int64_t k = gguf_find_key(gguf, "chatterbox_ve.final_relu");
        ve->cfg_.final_relu = (k < 0) ? true : (gguf_get_val_bool(gguf, k) != 0);
    }

    if (ve->cfg_.n_lstm_layers != 3) {
        std::fprintf(stderr,
                     "VE::load: only 3-layer LSTM supported, got %d\n",
                     ve->cfg_.n_lstm_layers);
        return nullptr;
    }

    auto get = [&](const char* name) -> ggml_tensor* {
        ggml_tensor* t = model->find_tensor(name);
        if (!t) std::fprintf(stderr, "VE::load: missing tensor %s\n", name);
        return t;
    };

    for (int l = 0; l < 3; ++l) {
        char nm[64];
        std::snprintf(nm, sizeof(nm), "lstm.l%d.weight_ih", l); ve->w_ih_[l] = get(nm);
        std::snprintf(nm, sizeof(nm), "lstm.l%d.weight_hh", l); ve->w_hh_[l] = get(nm);
        std::snprintf(nm, sizeof(nm), "lstm.l%d.bias_ih",   l); ve->b_ih_[l] = get(nm);
        std::snprintf(nm, sizeof(nm), "lstm.l%d.bias_hh",   l); ve->b_hh_[l] = get(nm);
        if (!ve->w_ih_[l] || !ve->w_hh_[l] || !ve->b_ih_[l] || !ve->b_hh_[l]) {
            return nullptr;
        }
    }
    ve->proj_w_ = get("proj.weight");
    ve->proj_b_ = get("proj.bias");
    if (!ve->proj_w_ || !ve->proj_b_) return nullptr;

    ve->backend_ = chatterbox::default_backend();
    if (!ve->backend_) {
        std::fprintf(stderr, "VE::load: ggml_backend_cpu_init failed\n");
        return nullptr;
    }
    return ve;
}

std::vector<float> VE::forward(const std::vector<float>& mel_flat,
                                int n_frames) {
    const int H = cfg_.hidden;
    const int M = cfg_.n_mels;
    const int E = cfg_.emb_dim;

    if (n_frames <= 0) {
        std::fprintf(stderr, "VE::forward: n_frames must be > 0\n");
        return {};
    }
    if (static_cast<int>(mel_flat.size()) != n_frames * M) {
        std::fprintf(stderr,
                     "VE::forward: mel buffer is %zu floats, expected %d*%d=%d\n",
                     mel_flat.size(), n_frames, M, n_frames * M);
        return {};
    }

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
        std::fprintf(stderr, "VE::forward: ggml_init failed\n");
        return {};
    }
    ggml_cgraph* gf = ggml_new_graph_custom(ctx, GRAPH_MAX_NODES, false);

    // Inputs: mel (M, T) fp32, plus zero-initialized h0/c0 for each layer.
    ggml_tensor* mel = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, M, n_frames);
    ggml_set_name(mel, "mel");
    ggml_set_input(mel);

    ggml_tensor* h_init[3];
    ggml_tensor* c_init[3];
    for (int l = 0; l < 3; ++l) {
        h_init[l] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
        c_init[l] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
        char nm[32];
        std::snprintf(nm, sizeof(nm), "h0_l%d", l); ggml_set_name(h_init[l], nm);
        std::snprintf(nm, sizeof(nm), "c0_l%d", l); ggml_set_name(c_init[l], nm);
        ggml_set_input(h_init[l]);
        ggml_set_input(c_init[l]);
    }

    // Per-layer combined bias = b_ih + b_hh (both fp32, computed once).
    ggml_tensor* bias_comb[3];
    for (int l = 0; l < 3; ++l) {
        bias_comb[l] = ggml_add(ctx, b_ih_[l], b_hh_[l]);
    }

    // For each LSTM layer:
    //   For each timestep t: build cell (gates -> activations -> c, h)
    //   Write h_t into h_seq[t]; h_seq becomes next layer's input.
    ggml_tensor* layer_input = mel;        // (M or H, T)
    ggml_tensor* last_h = nullptr;
    for (int l = 0; l < 3; ++l) {
        const int input_dim = (l == 0) ? M : H;

        // Per-layer output sequence h_seq is built as a pure DAG node:
        // each timestep's h is reshaped to (H, 1) and concatenated onto
        // the running h_seq along the time axis. This is essential for
        // ggml_backend_sched compatibility — sched needs every tensor
        // to be either a graph input (set externally) or a node with
        // operator-defined producers, not a leaf written to via
        // ggml_cpy. (The cpy-into-leaf pattern fails sched's split
        // analysis with buffer_id=-1 in ggml-alloc.)
        //
        // The cost: O(T) concats per layer, each of size O(H * T_so_far).
        // For our typical T<800 this is ~10 ms of host-side graph
        // building; the actual compute is dominated by the matmuls.
        ggml_tensor* h_seq = nullptr;

        ggml_tensor* h = h_init[l];
        ggml_tensor* c = c_init[l];

        for (int t = 0; t < n_frames; ++t) {
            // x_t = layer_input[:, t] as a 1D view of length input_dim.
            ggml_tensor* x_t = ggml_view_1d(ctx, layer_input,
                input_dim, t * layer_input->nb[1]);

            // gates = W_ih @ x_t + W_hh @ h + (b_ih + b_hh)   -> (4H,)
            ggml_tensor* in_t = ggml_mul_mat(ctx, w_ih_[l], x_t);  // (4H,)
            ggml_tensor* hi_t = ggml_mul_mat(ctx, w_hh_[l], h);    // (4H,)
            ggml_tensor* gates = ggml_add(ctx, ggml_add(ctx, in_t, hi_t),
                                                bias_comb[l]);

            // Slice gates into i / f / g / o views (each (H,)).
            const size_t e = gates->nb[0];
            ggml_tensor* i_pre = ggml_view_1d(ctx, gates, H, 0 * H * e);
            ggml_tensor* f_pre = ggml_view_1d(ctx, gates, H, 1 * H * e);
            ggml_tensor* g_pre = ggml_view_1d(ctx, gates, H, 2 * H * e);
            ggml_tensor* o_pre = ggml_view_1d(ctx, gates, H, 3 * H * e);

            ggml_tensor* i = ggml_sigmoid(ctx, i_pre);
            ggml_tensor* f = ggml_sigmoid(ctx, f_pre);
            ggml_tensor* g = ggml_tanh   (ctx, g_pre);
            ggml_tensor* o = ggml_sigmoid(ctx, o_pre);

            // c = f * c + i * g
            c = ggml_add(ctx, ggml_mul(ctx, f, c), ggml_mul(ctx, i, g));
            // h = o * tanh(c)
            h = ggml_mul(ctx, o, ggml_tanh(ctx, c));

            // Append h to h_seq along the time axis. Reshape (H,) ->
            // (H, 1) first so ggml_concat with dim=1 grows ne[1].
            ggml_tensor* h_col = ggml_reshape_2d(ctx, h, H, 1);
            h_seq = (h_seq == nullptr)
                ? h_col
                : ggml_concat(ctx, h_seq, h_col, /*dim=*/1);
        }

        // Make h_seq contiguous so the next layer's per-time view
        // arithmetic uses normal strides.
        h_seq = ggml_cont(ctx, h_seq);
        char snm[32]; std::snprintf(snm, sizeof(snm), "h_seq_l%d", l);
        ggml_set_name(h_seq, snm);

        layer_input = h_seq;
        if (l == 2) last_h = h;     // keep the very last hidden of the top layer
    }

    // Projection: y = proj_w @ last_h + proj_b. Linear-layout weight, so
    // proj_w has ggml ne[0]=in=256, ne[1]=out=256. No transpose needed.
    ggml_tensor* y = ggml_mul_mat(ctx, proj_w_, last_h);
    y = ggml_add(ctx, y, proj_b_);

    if (cfg_.final_relu) {
        y = ggml_relu(ctx, y);
    }
    ggml_set_name(y, "y_pre_norm");
    ggml_set_output(y);
    ggml_build_forward_expand(gf, y);

    ggml_backend_sched_t sched = chatterbox::make_sched(GRAPH_MAX_NODES);
    if (!sched || !ggml_backend_sched_alloc_graph(sched, gf)) {
        std::fprintf(stderr, "VE::forward: gallocr_alloc_graph failed\n");
        if (sched) ggml_backend_sched_free(sched);
        ggml_free(ctx);
        return {};
    }

    // Fill inputs.
    ggml_backend_tensor_set(mel, mel_flat.data(), 0,
                            mel_flat.size() * sizeof(float));
    std::vector<float> zeros(H, 0.0f);
    for (int l = 0; l < 3; ++l) {
        ggml_backend_tensor_set(h_init[l], zeros.data(), 0,
                                H * sizeof(float));
        ggml_backend_tensor_set(c_init[l], zeros.data(), 0,
                                H * sizeof(float));
    }

    if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "VE::forward: backend_graph_compute failed\n");
        if (sched) ggml_backend_sched_free(sched);
        ggml_free(ctx);
        return {};
    }

    std::vector<float> result(static_cast<size_t>(E));
    ggml_backend_tensor_get(y, result.data(), 0, result.size() * sizeof(float));

    if (sched) ggml_backend_sched_free(sched);
    ggml_free(ctx);

    // L2 normalize host-side. Doing this in the graph would need a
    // broadcasted scalar division op, which ggml supports but takes more
    // graph nodes for marginal benefit on a 256-d vector.
    double sumsq = 0.0;
    for (float v : result) sumsq += static_cast<double>(v) * v;
    const float nrm = static_cast<float>(std::sqrt(sumsq));
    if (nrm > 0.0f) {
        const float inv = 1.0f / nrm;
        for (float& v : result) v *= inv;
    }
    return result;
}

}  // namespace chatterbox
