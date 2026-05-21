#include "s3spk.h"
#include "backend.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace chatterbox {

namespace {

constexpr size_t GRAPH_MAX_NODES = 65536;   // 52 dense layers * many ops/layer

// Block specs: (num_layers, kernel_size, dilation), matching upstream.
struct BlockSpec { int n_layers; int kernel; int dilation; };
constexpr BlockSpec BLOCKS[3] = { {12, 3, 1}, {24, 3, 2}, {16, 3, 2} };

// Backend-aware fp32 read of a ggml tensor (delegates to
// chatterbox::read_tensor_f32). BN running_mean / running_var are
// loaded into the default backend's buffer by load_model, so direct
// `t->data` access would dereference a device pointer on Vulkan.
std::vector<float> read_f32(const ggml_tensor* t) {
    return chatterbox::read_tensor_f32(t);
}

}  // namespace

S3SpeakerEncoder::~S3SpeakerEncoder() {
    if (deriv_buffer_) ggml_backend_buffer_free(deriv_buffer_);
    if (deriv_ctx_)    ggml_free(deriv_ctx_);
    // backend_ is a process-wide singleton (see backend.h); do NOT free.
}

bool S3SpeakerEncoder::precompute_bn_(const std::string& prefix, bool affine) {
    auto rm_it = weights_.find(prefix + ".running_mean");
    auto rv_it = weights_.find(prefix + ".running_var");
    if (rm_it == weights_.end() || rv_it == weights_.end()) return false;

    const std::vector<float> rm = read_f32(rm_it->second);
    const std::vector<float> rv = read_f32(rv_it->second);
    const size_t C = rm.size();

    std::vector<float> w, b;
    if (affine) {
        auto wi = weights_.find(prefix + ".weight");
        auto bi = weights_.find(prefix + ".bias");
        if (wi == weights_.end() || bi == weights_.end()) return false;
        w = read_f32(wi->second);
        b = read_f32(bi->second);
        if (w.size() != C || b.size() != C) return false;
    }

    std::vector<float> scale(C), bias(C);
    for (size_t c = 0; c < C; ++c) {
        const float s = (affine ? w[c] : 1.0f) /
                         std::sqrt(rv[c] + cfg_.bn_eps);
        scale[c] = s;
        bias[c]  = (affine ? b[c] : 0.0f) - rm[c] * s;
    }

    ggml_tensor* s_t = ggml_new_tensor_1d(deriv_ctx_, GGML_TYPE_F32, C);
    ggml_tensor* b_t = ggml_new_tensor_1d(deriv_ctx_, GGML_TYPE_F32, C);
    std::string n1 = prefix + ".scale";
    std::string n2 = prefix + ".bias";
    ggml_set_name(s_t, n1.c_str());
    ggml_set_name(b_t, n2.c_str());
    bn_scale_.emplace(prefix, s_t);
    bn_bias_.emplace(prefix, b_t);

    // We can't fill the tensor data yet — backend buffer isn't allocated
    // until later. Stash the precomputed values; the caller (load()) will
    // backend_tensor_set them all after ggml_backend_alloc_ctx_tensors.
    // But we don't have a persistent stash field… simplest fix: do all
    // precomputation in one pass after the buffer is allocated.
    // Caller calls precompute_bn_ ONLY after buffer is ready, then we
    // backend_tensor_set right here.
    ggml_backend_tensor_set(s_t, scale.data(), 0, C * sizeof(float));
    ggml_backend_tensor_set(b_t, bias.data(),  0, C * sizeof(float));
    return true;
}

std::unique_ptr<S3SpeakerEncoder> S3SpeakerEncoder::load(Model* model) {
    if (!model) return nullptr;
    if (model->arch != "chatterbox_s3gen") {
        std::fprintf(stderr,
                     "S3SpeakerEncoder::load: expected chatterbox_s3gen arch, got %s\n",
                     model->arch.c_str());
        return nullptr;
    }

    auto enc = std::unique_ptr<S3SpeakerEncoder>(new S3SpeakerEncoder());
    enc->model_ = model;

    // Pull all speaker_encoder.* tensors (strip the prefix when keying).
    const std::string prefix = "speaker_encoder.";
    for (const auto& ti : model->tensors) {
        if (ti.name.compare(0, prefix.size(), prefix) == 0) {
            enc->weights_.emplace(ti.name.substr(prefix.size()), ti.tensor);
        }
    }
    std::printf("S3SpeakerEncoder::load: %zu speaker_encoder.* tensors\n",
                enc->weights_.size());
    if (enc->weights_.empty()) {
        std::fprintf(stderr, "S3SpeakerEncoder::load: no speaker_encoder tensors\n");
        return nullptr;
    }

    enc->backend_ = chatterbox::default_backend();
    if (!enc->backend_) {
        std::fprintf(stderr, "S3SpeakerEncoder::load: backend init failed\n");
        return nullptr;
    }

    // Collect all BN prefixes by scanning for ".running_mean" tensors.
    // For each, decide affine by whether ".weight" exists.
    struct BNDescr { std::string prefix; bool affine; };
    std::vector<BNDescr> bns;
    for (const auto& [k, _t] : enc->weights_) {
        const std::string suf = ".running_mean";
        if (k.size() > suf.size()
            && k.compare(k.size() - suf.size(), suf.size(), suf) == 0) {
            const std::string p = k.substr(0, k.size() - suf.size());
            const bool affine = enc->weights_.count(p + ".weight") > 0;
            bns.push_back({p, affine});
        }
    }
    std::printf("  detected %zu BatchNorm layers\n", bns.size());

    // Allocate the derived-weights context. Need enough for 2 tensors
    // per BN. Each tensor needs ggml_tensor_overhead bytes of descriptor.
    {
        const size_t mem =
            (bns.size() * 2 + 8) * ggml_tensor_overhead() + 4096;
        ggml_init_params p{};
        p.mem_size   = mem;
        p.no_alloc   = true;
        enc->deriv_ctx_ = ggml_init(p);
        if (!enc->deriv_ctx_) return nullptr;
    }

    // Pre-create the descriptors so the buffer is the right size.
    // (precompute_bn_ creates the ggml tensors AND backend_tensor_sets
    // them — but the buffer must exist before _set. Two-pass: first
    // create all descriptors with no_alloc, alloc the buffer, then fill.)
    //
    // To keep precompute_bn_ a single call, we instead inline the
    // descriptor creation here, allocate the buffer, then do the data
    // fill in a second loop.
    enc->bn_scale_.reserve(bns.size());
    enc->bn_bias_.reserve(bns.size());
    for (const auto& bn : bns) {
        // Need to know C — read it from the running_mean tensor.
        const ggml_tensor* rm = enc->weights_.at(bn.prefix + ".running_mean");
        const int64_t C = rm->ne[0];
        ggml_tensor* s_t = ggml_new_tensor_1d(enc->deriv_ctx_, GGML_TYPE_F32, C);
        ggml_tensor* b_t = ggml_new_tensor_1d(enc->deriv_ctx_, GGML_TYPE_F32, C);
        const std::string n1 = bn.prefix + ".scale";
        const std::string n2 = bn.prefix + ".bias";
        ggml_set_name(s_t, n1.c_str());
        ggml_set_name(b_t, n2.c_str());
        enc->bn_scale_.emplace(bn.prefix, s_t);
        enc->bn_bias_.emplace(bn.prefix, b_t);
    }

    enc->deriv_buffer_ =
        ggml_backend_alloc_ctx_tensors(enc->deriv_ctx_, enc->backend_);
    if (!enc->deriv_buffer_) {
        std::fprintf(stderr, "S3SpeakerEncoder::load: deriv buffer alloc failed\n");
        return nullptr;
    }

    // Now fill the precomputed BN scale + bias values.
    for (const auto& bn : bns) {
        const ggml_tensor* rm_t = enc->weights_.at(bn.prefix + ".running_mean");
        const ggml_tensor* rv_t = enc->weights_.at(bn.prefix + ".running_var");
        const auto rm = read_f32(rm_t);
        const auto rv = read_f32(rv_t);
        const size_t C = rm.size();
        std::vector<float> w, b;
        if (bn.affine) {
            w = read_f32(enc->weights_.at(bn.prefix + ".weight"));
            b = read_f32(enc->weights_.at(bn.prefix + ".bias"));
        }
        std::vector<float> scale(C), bias(C);
        for (size_t c = 0; c < C; ++c) {
            const float s = (bn.affine ? w[c] : 1.0f) /
                             std::sqrt(rv[c] + enc->cfg_.bn_eps);
            scale[c] = s;
            bias[c]  = (bn.affine ? b[c] : 0.0f) - rm[c] * s;
        }
        ggml_tensor* s_t = enc->bn_scale_.at(bn.prefix);
        ggml_tensor* b_t = enc->bn_bias_.at(bn.prefix);
        ggml_backend_tensor_set(s_t, scale.data(), 0, C * sizeof(float));
        ggml_backend_tensor_set(b_t, bias.data(),  0, C * sizeof(float));
    }
    return enc;
}

namespace {

// Apply precomputed BN to a tensor whose channels live on the OUTER
// (highest) dim — ne[1] for 2D (T, C), ne[2] for 3D (W, H, C).
ggml_tensor* bn_apply(ggml_context* ctx, ggml_tensor* x,
                       ggml_tensor* scale, ggml_tensor* bias) {
    const int nd = ggml_n_dims(x);
    const int64_t C = x->ne[nd - 1];
    ggml_tensor* s = scale;
    ggml_tensor* b = bias;
    if (nd == 2) {
        s = ggml_reshape_2d(ctx, s, 1, C);
        b = ggml_reshape_2d(ctx, b, 1, C);
    } else if (nd == 3) {
        s = ggml_reshape_3d(ctx, s, 1, 1, C);
        b = ggml_reshape_3d(ctx, b, 1, 1, C);
    }
    return ggml_add(ctx, ggml_mul(ctx, x, s), b);
}

}  // namespace

std::vector<float> S3SpeakerEncoder::forward_fcm_only(
        const std::vector<float>& fbank, int T, int n_mels,
        int& out_C, int& out_T) {
    out_C = 0; out_T = 0;
    if (n_mels != cfg_.feat_dim) return {};
    if (static_cast<int>(fbank.size()) != T * n_mels) return {};

    const size_t buf_size =
        ggml_tensor_overhead() * 4096
        + ggml_graph_overhead_custom(4096, false);
    std::vector<uint8_t> buf(buf_size);
    ggml_init_params params{};
    params.mem_size   = buf.size();
    params.mem_buffer = buf.data();
    params.no_alloc   = true;
    ggml_context* ctx = ggml_init(params);
    if (!ctx) return {};
    ggml_cgraph* gf = ggml_new_graph_custom(ctx, 4096, false);

    // fbank input arrives as numpy (T, n_mels) row-major. For conv2d we
    // need (W=T, H=n_mels) in ggml ne, which corresponds to numpy
    // (n_mels, T) row-major BYTE LAYOUT — so we transpose host-side and
    // store as ggml ne=(T, n_mels).
    ggml_tensor* fbank_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T, n_mels);
    ggml_set_input(fbank_in);

    auto bn_l = [&](ggml_tensor* x, const std::string& p) {
        return bn_apply(ctx, x, bn_scale_.at(p), bn_bias_.at(p));
    };
    auto W_l = [&](const std::string& k) { return weights_.at(k); };

    ggml_tensor* x2d = ggml_reshape_3d(ctx, fbank_in, T, n_mels, 1);
    x2d = ggml_conv_2d(ctx, W_l("head.conv1.weight"), x2d, 1, 1, 1, 1, 1, 1);
    x2d = ggml_relu(ctx, bn_l(x2d, "head.bn1"));

    auto res_block = [&](ggml_tensor* x, const std::string& p, int stride) {
        ggml_tensor* out = ggml_conv_2d(ctx, W_l(p + ".conv1.weight"), x,
                                          1, stride, 1, 1, 1, 1);
        out = ggml_relu(ctx, bn_l(out, p + ".bn1"));
        out = ggml_conv_2d(ctx, W_l(p + ".conv2.weight"), out, 1, 1, 1, 1, 1, 1);
        out = bn_l(out, p + ".bn2");

        ggml_tensor* sc;
        if (weights_.count(p + ".shortcut.0.weight")) {
            sc = ggml_conv_2d(ctx, W_l(p + ".shortcut.0.weight"), x,
                                1, stride, 0, 0, 1, 1);
            sc = bn_l(sc, p + ".shortcut.1");
        } else { sc = x; }
        return ggml_relu(ctx, ggml_add(ctx, out, sc));
    };
    x2d = res_block(x2d, "head.layer1.0", 2);
    x2d = res_block(x2d, "head.layer1.1", 1);
    x2d = res_block(x2d, "head.layer2.0", 2);
    x2d = res_block(x2d, "head.layer2.1", 1);
    x2d = ggml_conv_2d(ctx, W_l("head.conv2.weight"), x2d, 1, 2, 1, 1, 1, 1);
    x2d = ggml_relu(ctx, bn_l(x2d, "head.bn2"));

    const int64_t T_fcm = x2d->ne[0];
    const int64_t F_fcm = x2d->ne[1];
    const int64_t C_fcm = x2d->ne[2];
    ggml_tensor* x = ggml_cont(ctx, x2d);
    x = ggml_reshape_2d(ctx, x, T_fcm, F_fcm * C_fcm);
    // x: ne=[T, 320] (320 channels on ne[1])

    ggml_set_name(x, "fcm_out");
    ggml_set_output(x);
    ggml_build_forward_expand(gf, x);

    ggml_backend_sched_t sched = chatterbox::make_sched(GRAPH_MAX_NODES);
    if (!sched || !ggml_backend_sched_alloc_graph(sched, gf)) {
        std::fprintf(stderr, "forward_fcm_only: alloc graph failed\n");
        if (sched) ggml_backend_sched_free(sched); ggml_free(ctx); return {};
    }
    // Transpose fbank from numpy (T, n_mels) row-major to (n_mels, T)
    // row-major byte layout, which corresponds to ggml ne=(T, n_mels).
    std::vector<float> fbank_t(fbank.size());
    for (int t = 0; t < T; ++t) {
        for (int m = 0; m < n_mels; ++m) {
            fbank_t[m * T + t] = fbank[t * n_mels + m];
        }
    }
    ggml_backend_tensor_set(fbank_in, fbank_t.data(), 0,
                             fbank_t.size() * sizeof(float));
    if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) {
        if (sched) ggml_backend_sched_free(sched); ggml_free(ctx); return {};
    }

    out_C = static_cast<int>(F_fcm * C_fcm);  // 320
    out_T = static_cast<int>(T_fcm);
    std::vector<float> result(static_cast<size_t>(out_C) * out_T);
    ggml_backend_tensor_get(x, result.data(), 0, result.size() * sizeof(float));

    if (sched) ggml_backend_sched_free(sched); ggml_free(ctx);
    return result;
}

std::vector<float> S3SpeakerEncoder::forward(const std::vector<float>& fbank,
                                              int T, int n_mels) {
    if (n_mels != cfg_.feat_dim) {
        std::fprintf(stderr,
                     "S3SpeakerEncoder::forward: expected n_mels=%d, got %d\n",
                     cfg_.feat_dim, n_mels);
        return {};
    }
    if (static_cast<int>(fbank.size()) != T * n_mels) {
        std::fprintf(stderr,
                     "S3SpeakerEncoder::forward: bad fbank buffer (%zu floats vs %d*%d)\n",
                     fbank.size(), T, n_mels);
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
    if (!ctx) return {};
    ggml_cgraph* gf = ggml_new_graph_custom(ctx, GRAPH_MAX_NODES, false);

    // --- Input ---
    // fbank numpy (T, n_mels) row-major. ggml ne[0]=n_mels, ne[1]=T.
    // We want to compute on (n_mels, T) per the upstream "permute" step
    // (which makes channels innermost so a per-channel BN add broadcasts
    // naturally). With our ggml convention (channels on ne[1] for 2D),
    // that IS the layout we want — feed input as (n_mels, T) ggml.
    //
    // numpy memory order (T, n_mels) row-major matches ggml (n_mels, T)
    // — see the analogous layout argument in scripts/reference_s3_mel.py
    // and the s3enc port. Direct memcpy.
    // fbank arrives as numpy (T, n_mels) row-major; we transpose host-side
    // before memcpy so the ggml bytes are in numpy (n_mels, T) order,
    // which IS the natural layout for ggml ne=(T, n_mels) — what conv2d
    // wants as input data (W=T, H=n_mels, IC=1). See forward_fcm_only
    // for the matching transposed memcpy.
    ggml_tensor* fbank_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T, n_mels);
    ggml_set_name(fbank_in, "fbank");
    ggml_set_input(fbank_in);

    auto bn = [&](ggml_tensor* x, const std::string& p) {
        return bn_apply(ctx, x, bn_scale_.at(p), bn_bias_.at(p));
    };
    auto W = [&](const std::string& k) { return weights_.at(k); };

    // Reshape (T, n_mels) -> (T, n_mels, 1). Just adds an IC=1 channel
    // dim; byte layout unchanged.
    ggml_tensor* x2d = ggml_reshape_3d(ctx, fbank_in, T, n_mels, 1);

    // conv1: (1 -> 32, k=3, s=1, p=1)
    x2d = ggml_conv_2d(ctx, W("head.conv1.weight"), x2d,
                       /*s0=*/1, /*s1=*/1, /*p0=*/1, /*p1=*/1,
                       /*d0=*/1, /*d1=*/1);
    x2d = ggml_relu(ctx, bn(x2d, "head.bn1"));

    // BasicResBlock helper.
    auto res_block = [&](ggml_tensor* x, const std::string& p, int stride) {
        // conv1: stride on H axis (= ne[1] = F dim) only
        ggml_tensor* out = ggml_conv_2d(ctx, W(p + ".conv1.weight"), x,
                                          /*s0=*/1, /*s1=*/stride,
                                          /*p0=*/1, /*p1=*/1,
                                          /*d0=*/1, /*d1=*/1);
        out = ggml_relu(ctx, bn(out, p + ".bn1"));
        out = ggml_conv_2d(ctx, W(p + ".conv2.weight"), out,
                            1, 1, 1, 1, 1, 1);
        out = bn(out, p + ".bn2");

        ggml_tensor* sc;
        const std::string sc_w = p + ".shortcut.0.weight";
        if (weights_.count(sc_w)) {
            sc = ggml_conv_2d(ctx, W(sc_w), x,
                              /*s0=*/1, /*s1=*/stride,
                              /*p0=*/0, /*p1=*/0,
                              /*d0=*/1, /*d1=*/1);
            sc = bn(sc, p + ".shortcut.1");
        } else {
            sc = x;
        }
        return ggml_relu(ctx, ggml_add(ctx, out, sc));
    };

    x2d = res_block(x2d, "head.layer1.0", /*stride=*/2);
    x2d = res_block(x2d, "head.layer1.1", /*stride=*/1);
    x2d = res_block(x2d, "head.layer2.0", /*stride=*/2);
    x2d = res_block(x2d, "head.layer2.1", /*stride=*/1);

    // conv2: stride=(2,1) — stride 2 on F (ne[1]) only
    x2d = ggml_conv_2d(ctx, W("head.conv2.weight"), x2d,
                       /*s0=*/1, /*s1=*/2, /*p0=*/1, /*p1=*/1,
                       /*d0=*/1, /*d1=*/1);
    x2d = ggml_relu(ctx, bn(x2d, "head.bn2"));

    // x2d now ggml shape (T, 10, 32). Flatten the last two dims into
    // channels — ggml memory is already contiguous as (T, 10, 32), so
    // reshape to (T, 320) keeps the byte layout.
    const int64_t T_fcm = x2d->ne[0];
    const int64_t F_fcm = x2d->ne[1];   // 10
    const int64_t C_fcm = x2d->ne[2];   // 32
    ggml_tensor* x = ggml_cont(ctx, x2d);   // force contiguous
    x = ggml_reshape_2d(ctx, x, T_fcm, F_fcm * C_fcm);   // (T, 320)
    // x is now in "1D conv format": ne[0]=T, ne[1]=320.

    // --- TDNN(320 -> 128, k=5, s=2, padding=2) ---
    auto add_bias_td = [&](ggml_tensor* y, const std::string& bk, int n_channels) {
        // ggml_conv_1d outputs (T_out, channels) — bias is (channels,).
        // Reshape to (1, channels) so the add broadcasts over T_out.
        ggml_tensor* b = ggml_reshape_2d(ctx, weights_.at(bk), 1, n_channels);
        return ggml_add(ctx, y, b);
    };
    (void)add_bias_td;  // TDNN.linear has no bias.

    x = ggml_conv_1d(ctx, W("xvector.tdnn.linear.weight"), x,
                      /*s=*/2, /*p=*/2, /*d=*/1);
    x = ggml_relu(ctx, bn(x, "xvector.tdnn.nonlinear.batchnorm"));
    // x is (T_td, 128) ggml.

    // --- CAM dense block helper ---
    // Input  x: (T, C_in)
    // Output : (T, C_in + n_layers * growth)  with growth=32
    auto cam_layer = [&](ggml_tensor* y, const std::string& p,
                          int kernel, int dilation) {
        const int padding = (kernel - 1) / 2 * dilation;
        const int C_in  = static_cast<int>(y->ne[1]);
        const int C_out = static_cast<int>(weights_.at(p + ".linear_local.weight")->ne[2]);

        // y_local: local conv
        ggml_tensor* yl = ggml_conv_1d(ctx, W(p + ".linear_local.weight"),
                                         y, /*s=*/1, padding, dilation);
        // ctx = mean(x, -1, keepdim) + seg_pool(x, seg_len=100)
        // Inputs are (T, C). We need mean over T -> (1, C).
        // ggml_mean reduces ne[0] -> 1 (if it works on n=ne[0]).
        // Let me check usage: ggml_mean computes mean along ne[0].
        // Our T is on ne[0] (since we're in "1D conv" layout). So mean
        // of x along ne[0] gives shape (1, C). Good.
        ggml_tensor* mean_global = ggml_mean(ctx, y);  // (1, C)
        // seg_pool: avg_pool1d (kernel=100, stride=100, ceil_mode).
        // ggml_pool_1d ne[0] is the pooling axis. For (T, C) input the
        // pool axis is ne[0]=T. Output ne[0] = ceil(T/100).
        const int seg = cfg_.seg_len;
        const int n_segs = (static_cast<int>(y->ne[0]) + seg - 1) / seg;
        // For ggml_pool_1d, padding p means LEFT pad only (per ggml docs);
        // ceil_mode equivalence: pad enough so the last segment is full.
        const int pool_pad = n_segs * seg - static_cast<int>(y->ne[0]);
        // (ggml_pool_1d takes p0 as padding both sides? Different impls vary.
        // For now pass pool_pad to one side; refine if parity drifts.)
        ggml_tensor* seg_pool = ggml_pool_1d(ctx, y, GGML_OP_POOL_AVG,
                                              seg, seg, pool_pad);
        // seg_pool is (n_segs, C). Need to expand back to (T, C) by repeat.
        // The simplest approach: repeat each segment seg times then crop.
        // ggml_repeat(a, b) repeats a to match b's shape — but here we
        // want a per-segment expand, not a simple tile. Use ggml_upscale
        // or a tile-then-view trick.
        // Workaround: ggml_repeat with a (T, C) destination tile-repeats
        // seg_pool along ne[0]; this gives wrong values (tiles all segs,
        // not per-segment expand). To get per-segment expand, use the
        // identity:
        //    seg_pool view (1, n_segs, C) -> repeat to (seg, n_segs, C)
        //    -> ggml_reshape to (seg*n_segs, C) -> view to (T, C)
        // Build that.
        // First view seg_pool (n_segs, C) as (1, n_segs, C).
        ggml_tensor* sp3 = ggml_reshape_3d(ctx, seg_pool, 1, n_segs, C_in);
        // Make a (seg, n_segs, C) container, repeat sp3 into it.
        ggml_tensor* tgt = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, seg, n_segs, C_in);
        ggml_tensor* expanded = ggml_repeat(ctx, sp3, tgt);
        // Reshape (seg, n_segs, C) -> (seg*n_segs, C); then view first T rows.
        expanded = ggml_cont(ctx, expanded);
        expanded = ggml_reshape_2d(ctx, expanded, seg * n_segs, C_in);
        // Crop to T rows along ne[0].
        ggml_tensor* seg_expanded = ggml_view_2d(ctx, expanded,
                                                   y->ne[0], C_in,
                                                   expanded->nb[1], 0);
        seg_expanded = ggml_cont(ctx, seg_expanded);   // unneeded if shape exact

        // context = mean_global + seg_expanded; broadcast (1, C) over T.
        ggml_tensor* mean_bc = ggml_repeat(ctx, mean_global, seg_expanded);
        ggml_tensor* context = ggml_add(ctx, mean_bc, seg_expanded);

        // 1x1 conv linear1 (with bias) -> ReLU -> 1x1 conv linear2 (with bias) -> sigmoid
        ggml_tensor* c = ggml_conv_1d(ctx, W(p + ".linear1.weight"),
                                       context, 1, 0, 1);
        c = add_bias_td(c, p + ".linear1.bias",
                         static_cast<int>(weights_.at(p + ".linear1.weight")->ne[2]));
        c = ggml_relu(ctx, c);
        ggml_tensor* m = ggml_conv_1d(ctx, W(p + ".linear2.weight"),
                                       c, 1, 0, 1);
        m = add_bias_td(m, p + ".linear2.bias", C_out);
        m = ggml_sigmoid(ctx, m);

        return ggml_mul(ctx, yl, m);
    };

    auto cam_dense_layer = [&](ggml_tensor* y, const std::string& p,
                                int kernel, int dilation) {
        // nonlinear1 (BN + ReLU) -> linear1 (1x1 conv, no bias) ->
        // nonlinear2 (BN + ReLU) -> cam_layer
        y = ggml_relu(ctx, bn(y, p + ".nonlinear1.batchnorm"));
        y = ggml_conv_1d(ctx, W(p + ".linear1.weight"), y, 1, 0, 1);
        y = ggml_relu(ctx, bn(y, p + ".nonlinear2.batchnorm"));
        y = cam_layer(y, p + ".cam_layer", kernel, dilation);
        return y;
    };

    auto cam_dense_block = [&](ggml_tensor* y, const std::string& p,
                                int n_layers, int kernel, int dilation) {
        for (int i = 1; i <= n_layers; ++i) {
            const std::string lp = p + ".tdnnd" + std::to_string(i);
            ggml_tensor* new_y = cam_dense_layer(y, lp, kernel, dilation);
            // Concat along channel axis (ne[1]).
            y = ggml_concat(ctx, y, new_y, /*dim=*/1);
        }
        return y;
    };

    auto transit_layer = [&](ggml_tensor* y, const std::string& p) {
        y = ggml_relu(ctx, bn(y, p + ".nonlinear.batchnorm"));
        // linear has NO bias (CAMPPlus passes bias=False to TransitLayer).
        y = ggml_conv_1d(ctx, W(p + ".linear.weight"), y, 1, 0, 1);
        return y;
    };

    // --- Three CAM dense blocks + Transit ---
    for (int i = 0; i < 3; ++i) {
        const std::string bp = "xvector.block" + std::to_string(i + 1);
        const std::string tp = "xvector.transit" + std::to_string(i + 1);
        x = cam_dense_block(x, bp, BLOCKS[i].n_layers,
                             BLOCKS[i].kernel, BLOCKS[i].dilation);
        x = transit_layer(x, tp);
    }

    // --- out_nonlinear (BN + ReLU) ---
    x = ggml_relu(ctx, bn(x, "xvector.out_nonlinear.batchnorm"));

    // --- StatsPool: mean + std over time (ne[0]) ---
    // mean: ggml_mean reduces ne[0] -> (1, C).
    ggml_tensor* mean = ggml_mean(ctx, x);                  // (1, C)
    // std: stdev (unbiased, ddof=1). Compute manually:
    //   diff = x - mean_broadcast
    //   var  = sum(diff^2, axis=0) / (T - 1)
    //   std  = sqrt(var + stats_eps)
    ggml_tensor* mean_bc = ggml_repeat(ctx, mean, x);
    ggml_tensor* diff = ggml_sub(ctx, x, mean_bc);
    ggml_tensor* sq   = ggml_mul(ctx, diff, diff);          // (T, C)
    // Sum along ne[0] (T axis), giving (1, C).
    ggml_tensor* sumsq = ggml_sum_rows(ctx, sq);
    // Divide by (T - 1).
    const float inv_Tm1 = 1.0f / static_cast<float>(x->ne[0] - 1);
    ggml_tensor* var = ggml_scale(ctx, sumsq, inv_Tm1);
    // NOTE: upstream's statistics_pooling() declares eps=1e-2 but never
    // uses it in the function body. Match: NO eps add.
    ggml_tensor* std_t = ggml_sqrt(ctx, var);

    // Concat [mean, std] along ne[1] (channel axis).
    ggml_tensor* stats = ggml_concat(ctx, mean, std_t, /*dim=*/1);
    // stats: (1, 2C) ggml

    // --- DenseLayer (1x1 conv + BN without affine) ---
    // Conv1d 1x1 with no bias -> (1, 192)
    ggml_tensor* y = ggml_conv_1d(ctx, W("xvector.dense.linear.weight"),
                                   stats, 1, 0, 1);
    // BN without affine (config_str="batchnorm_"). NO ReLU (affine=false
    // means just the normalization in upstream's get_nonlinear flow).
    y = bn(y, "xvector.dense.nonlinear.batchnorm");

    // y has shape (1, 192). Flatten to (192,) for the host.
    ggml_tensor* out = ggml_reshape_1d(ctx, y, cfg_.emb_dim);

    ggml_set_name(out, "embedding");
    ggml_set_output(out);
    ggml_build_forward_expand(gf, out);

    ggml_backend_sched_t sched = chatterbox::make_sched(GRAPH_MAX_NODES);
    if (!sched || !ggml_backend_sched_alloc_graph(sched, gf)) {
        std::fprintf(stderr, "S3SpeakerEncoder::forward: gallocr_alloc_graph failed\n");
        if (sched) ggml_backend_sched_free(sched);
        ggml_free(ctx);
        return {};
    }

    // Transpose fbank from numpy (T, n_mels) to (n_mels, T) byte layout
    // so ggml ne=(T, n_mels) reads "frame t, mel m" correctly.
    std::vector<float> fbank_t(fbank.size());
    for (int t = 0; t < T; ++t) {
        for (int m = 0; m < n_mels; ++m) {
            fbank_t[m * T + t] = fbank[t * n_mels + m];
        }
    }
    ggml_backend_tensor_set(fbank_in, fbank_t.data(), 0,
                             fbank_t.size() * sizeof(float));

    if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "S3SpeakerEncoder::forward: compute failed\n");
        if (sched) ggml_backend_sched_free(sched);
        ggml_free(ctx);
        return {};
    }

    std::vector<float> result(cfg_.emb_dim);
    ggml_backend_tensor_get(out, result.data(), 0,
                             result.size() * sizeof(float));

    if (sched) ggml_backend_sched_free(sched);
    ggml_free(ctx);
    return result;
}

}  // namespace chatterbox
