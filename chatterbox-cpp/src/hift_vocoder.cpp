#include "hift_vocoder.h"
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
#include <random>
#include <string>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace chatterbox {

namespace {

constexpr size_t GRAPH_MAX_NODES = 131072;

// Backend-aware fp32 read of a ggml tensor. Delegates to
// chatterbox::read_tensor_f32 (ggml_backend_tensor_get) so it works
// when weights live in Vulkan memory.
std::vector<float> read_f32(const ggml_tensor* t) {
    return chatterbox::read_tensor_f32(t);
}

// mul_mat with fp32 accumulator on Vulkan. See cfm_decoder.cpp for the
// reason — fp16 matmul accumulation across the conv/F0 stack drifts
// past parity tolerances.
ggml_tensor* mm_f32(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b) {
    ggml_tensor* y = ggml_mul_mat(ctx, a, b);
    ggml_mul_mat_set_prec(y, GGML_PREC_F32);
    return y;
}

// conv1d via inlined im2col + fp32-acc mul_mat. Same shape contract as
// ggml_conv_1d (returns (OL, OC, N) ne layout).
ggml_tensor* conv_1d_f32(ggml_context* ctx,
                            ggml_tensor* w, ggml_tensor* x,
                            int s0, int p0, int d0) {
    ggml_tensor* im2col = ggml_im2col(ctx, w, x, s0, 0, p0, 0, d0, 0,
                                          false, GGML_TYPE_F16);
    ggml_tensor* im2col_2d = ggml_reshape_2d(ctx, im2col,
                                                  im2col->ne[0],
                                                  im2col->ne[1] * im2col->ne[2]);
    ggml_tensor* w_2d = ggml_reshape_2d(ctx, w,
                                            w->ne[0] * w->ne[1], w->ne[2]);
    ggml_tensor* y = mm_f32(ctx, im2col_2d, w_2d);   // (OL*N, OC)
    return ggml_reshape_3d(ctx, y,
                              im2col->ne[1], w->ne[2], im2col->ne[2]);
}

// Snake activation: y = x + (1/(alpha+eps)) * sin^2(x * alpha).
// Both alpha and alpha_inv are (C,) tensors; we reshape to (1, C) for
// broadcast across the time axis. Layout: x is (C, T) ne[0]=C, ne[1]=T.
ggml_tensor* snake(ggml_context* ctx, ggml_tensor* x,
                     ggml_tensor* alpha_1D, ggml_tensor* alpha_inv_1D) {
    // alpha_1D and alpha_inv_1D have ne=(C,). They broadcast over T
    // when used in element-wise ops with x of ne=(C, T).
    ggml_tensor* xa  = ggml_mul(ctx, x, alpha_1D);
    ggml_tensor* s   = ggml_sin(ctx, xa);
    ggml_tensor* s2  = ggml_sqr(ctx, s);
    ggml_tensor* mul = ggml_mul(ctx, s2, alpha_inv_1D);
    return ggml_add(ctx, x, mul);
}

// 1D conv on (C, T) layout. Transposes to (T, C) for ggml_conv_1d, then
// back. weight ne = (K, C_in, C_out). Bias broadcasts on the C_out axis.
ggml_tensor* conv1d_CT(ggml_context* ctx, ggml_tensor* x_CT,
                         const HiFTVocoder::ConvW& W,
                         int stride, int padding, int dilation,
                         int C_out) {
    // (C, T) -> (T, C)
    ggml_tensor* x_TC = ggml_cont(ctx, ggml_transpose(ctx, x_CT));
    // Conv1d output: (T_out, C_out)
    ggml_tensor* y_TC = conv_1d_f32(ctx, W.w, x_TC, stride, padding, dilation);
    if (W.b) {
        y_TC = ggml_add(ctx, y_TC, ggml_reshape_2d(ctx, W.b, 1, C_out));
    }
    return ggml_cont(ctx, ggml_transpose(ctx, y_TC));
}

// 1D transposed conv on (C, T) layout. ggml_conv_transpose_1d requires
// p=0, d=1, so we strip the upstream symmetric padding via view after.
//
// kernel weight ne = (K, C_out, C_in)  (note middle dim is C_out for
// conv_transpose — matches torch ConvTranspose1d weight ordering).
ggml_tensor* conv_transpose1d_CT(ggml_context* ctx, ggml_tensor* x_CT,
                                   const HiFTVocoder::ConvW& W,
                                   int stride, int padding, int C_out) {
    // ggml_conv_transpose_1d takes input as (T, C_in) ne[0]=T.
    ggml_tensor* x_TC = ggml_cont(ctx, ggml_transpose(ctx, x_CT));
    // Output: (T_full, C_out) where T_full = (T_in - 1) * stride + K.
    ggml_tensor* y_TC_full = ggml_conv_transpose_1d(ctx, W.w, x_TC, stride, 0, 1);
    // Strip symmetric padding along the T axis.
    const int T_full = static_cast<int>(y_TC_full->ne[0]);
    const int T_keep = T_full - 2 * padding;
    ggml_tensor* y_TC_view = ggml_view_2d(ctx, y_TC_full,
                                              T_keep, y_TC_full->ne[1],
                                              y_TC_full->nb[1],
                                              padding * sizeof(float));
    ggml_tensor* y_TC = ggml_cont(ctx, y_TC_view);
    if (W.b) {
        y_TC = ggml_add(ctx, y_TC, ggml_reshape_2d(ctx, W.b, 1, C_out));
    }
    return ggml_cont(ctx, ggml_transpose(ctx, y_TC));
}

// ResBlock forward. Three (Snake -> Conv k=K d=d_i -> Snake -> Conv k=K d=1 -> +x)
// layers in series. Each layer adds its output to the running residual.
ggml_tensor* resblock_forward(ggml_context* ctx, ggml_tensor* x_CT,
                                const HiFTVocoder::ResBlock& RB,
                                int channels) {
    const int dilations[3] = {1, 3, 5};
    const int K = RB.kernel_size;
    for (int i = 0; i < 3; ++i) {
        const auto& S = RB.sublayers[i];
        const int d = dilations[i];
        // padding for symmetric "same" output: (K-1)/2 * dilation
        const int p_d = (K - 1) / 2 * d;
        const int p_1 = (K - 1) / 2;

        ggml_tensor* h = snake(ctx, x_CT, S.alpha1, S.alpha1_inv);
        h = conv1d_CT(ctx, h, S.conv1, 1, p_d, d, channels);
        h = snake(ctx, h, S.alpha2, S.alpha2_inv);
        h = conv1d_CT(ctx, h, S.conv2, 1, p_1, 1, channels);
        x_CT = ggml_add(ctx, x_CT, h);
    }
    return x_CT;
}

}  // namespace


// ----------------------------------------------------------------------------
// Loader
// ----------------------------------------------------------------------------

HiFTVocoder::~HiFTVocoder() {
    if (deriv_buffer_) ggml_backend_buffer_free(deriv_buffer_);
    if (deriv_ctx_)    ggml_free(deriv_ctx_);
    // backend_ is a process-wide singleton (see backend.h); do NOT free.
}

namespace {

// Build a fp32 fused weight from weight_norm parametrization. Returns
// shape (K * C_in * C_out,) as a flat row-major vector with the original
// numpy (C_out, C_in, K) shape (i.e., g and v from torch).
std::vector<float> fuse_weight_norm_to_numpy(const ggml_tensor* g_t,
                                                const ggml_tensor* v_t) {
    // g_t numpy shape: (C_out, 1, 1) -> ggml ne=(1, 1, C_out, 1)
    // v_t numpy shape: (C_out, C_in, K) -> ggml ne=(K, C_in, C_out, 1)
    const int K = static_cast<int>(v_t->ne[0]);
    const int Cin = static_cast<int>(v_t->ne[1]);
    const int Cout = static_cast<int>(v_t->ne[2]);
    auto g = read_f32(g_t);                            // length C_out
    auto v = read_f32(v_t);                            // length K*Cin*Cout

    // norm[c_out] = sqrt(sum over (c_in, k) of v[c_out, c_in, k]^2)
    std::vector<float> norm(Cout, 0.0f);
    // v memory in ggml ne=(K, Cin, Cout): v_idx(k, c_in, c_out) = k + c_in*K + c_out*K*Cin
    for (int c_out = 0; c_out < Cout; ++c_out) {
        double acc = 0.0;
        for (int c_in = 0; c_in < Cin; ++c_in) {
            for (int k = 0; k < K; ++k) {
                const float val = v[k + c_in * K + c_out * K * Cin];
                acc += static_cast<double>(val) * val;
            }
        }
        norm[c_out] = static_cast<float>(std::sqrt(acc));
    }
    // Fuse: w[c_out, c_in, k] = g[c_out] * v[c_out, c_in, k] / norm[c_out].
    std::vector<float> out(static_cast<size_t>(K) * Cin * Cout);
    for (int c_out = 0; c_out < Cout; ++c_out) {
        const float gn = g[c_out] / std::max(norm[c_out], 1e-12f);
        for (int c_in = 0; c_in < Cin; ++c_in) {
            for (int k = 0; k < K; ++k) {
                const size_t idx = static_cast<size_t>(k) + c_in * K + c_out * K * Cin;
                out[idx] = gn * v[idx];
            }
        }
    }
    return out;
}

}  // namespace

bool HiFTVocoder::fetch_(Model* m) {
    auto get = [&](const std::string& name) -> ggml_tensor* {
        ggml_tensor* t = m->find_tensor(name);
        if (!t) std::fprintf(stderr, "HiFTVocoder::load: missing %s\n",
                              name.c_str());
        return t;
    };

    // Stage 1: collect all tensor pointers we need from the model
    // (these live in the GGUF buffer; weight_norm is two tensors per conv).
    struct ConvSrc {
        ggml_tensor* g = nullptr;     // original0 (1, 1, C_out)
        ggml_tensor* v = nullptr;     // original1 (K, C_in, C_out)
        ggml_tensor* bias = nullptr;  // .bias (C_out,)
        // For source_downs (NOT weight-normed) and similar, plain weight
        // is stored under .weight.
        ggml_tensor* w_plain = nullptr;
    };

    auto load_conv_src = [&](const std::string& p, ConvSrc& s) {
        const std::string g_key = p + ".parametrizations.weight.original0";
        const std::string v_key = p + ".parametrizations.weight.original1";
        const std::string w_key = p + ".weight";
        if (auto* g = m->find_tensor(g_key)) {
            s.g = g;
            s.v = get(v_key);
            s.bias = m->find_tensor(p + ".bias");
            return s.v != nullptr;
        }
        s.w_plain = m->find_tensor(w_key);
        if (s.w_plain) {
            s.bias = m->find_tensor(p + ".bias");
            return true;
        }
        std::fprintf(stderr, "HiFTVocoder::load: conv not found at %s\n",
                      p.c_str());
        return false;
    };

    // Gather all conv sources we need to fuse.
    struct ToFuse {
        std::string  name;          // for diagnostics
        ConvSrc      src;
        ConvW*       dst;
    };
    std::vector<ToFuse> all_convs;

    ConvSrc conv_pre_src, conv_post_src;
    if (!load_conv_src("mel2wav.conv_pre",  conv_pre_src))  return false;
    if (!load_conv_src("mel2wav.conv_post", conv_post_src)) return false;
    all_convs.push_back({"conv_pre",  conv_pre_src,  &conv_pre_});
    all_convs.push_back({"conv_post", conv_post_src, &conv_post_});

    ups_.resize(3);
    source_downs_.resize(3);
    for (int i = 0; i < 3; ++i) {
        ConvSrc us, sds;
        char nm[64];
        std::snprintf(nm, sizeof(nm), "mel2wav.ups.%d", i);
        if (!load_conv_src(nm, us)) return false;
        std::snprintf(nm, sizeof(nm), "mel2wav.source_downs.%d", i);
        if (!load_conv_src(nm, sds)) return false;
        all_convs.push_back({std::string("ups.") + std::to_string(i), us, &ups_[i]});
        all_convs.push_back({std::string("source_downs.") + std::to_string(i),
                              sds, &source_downs_[i]});
    }

    auto load_resblock = [&](ResBlock& RB, const std::string& p,
                                int kernel_size,
                                std::vector<ToFuse>& sink) {
        RB.kernel_size = kernel_size;
        RB.sublayers.resize(3);
        for (int i = 0; i < 3; ++i) {
            char nm[160];
            std::snprintf(nm, sizeof(nm), "%s.convs1.%d", p.c_str(), i);
            ConvSrc c1, c2;
            if (!load_conv_src(nm, c1)) return false;
            std::snprintf(nm, sizeof(nm), "%s.convs2.%d", p.c_str(), i);
            if (!load_conv_src(nm, c2)) return false;
            sink.push_back({p + ".convs1." + std::to_string(i), c1,
                              &RB.sublayers[i].conv1});
            sink.push_back({p + ".convs2." + std::to_string(i), c2,
                              &RB.sublayers[i].conv2});
            // Pull alpha tensors directly; we precompute inv_alpha below.
            std::snprintf(nm, sizeof(nm), "%s.activations1.%d.alpha", p.c_str(), i);
            RB.sublayers[i].alpha1 = get(nm);
            std::snprintf(nm, sizeof(nm), "%s.activations2.%d.alpha", p.c_str(), i);
            RB.sublayers[i].alpha2 = get(nm);
            if (!RB.sublayers[i].alpha1 || !RB.sublayers[i].alpha2) return false;
        }
        return true;
    };

    source_resblocks_.resize(3);
    for (int i = 0; i < 3; ++i) {
        char nm[64];
        std::snprintf(nm, sizeof(nm), "mel2wav.source_resblocks.%d", i);
        if (!load_resblock(source_resblocks_[i], nm,
                              cfg_.src_kernels[i], all_convs)) return false;
    }
    resblocks_.resize(9);
    for (int i = 0; i < 9; ++i) {
        char nm[64];
        std::snprintf(nm, sizeof(nm), "mel2wav.resblocks.%d", i);
        const int stage = i / 3;
        const int kid   = i % 3;
        if (!load_resblock(resblocks_[i], nm,
                              cfg_.resblock_kernels[kid], all_convs)) return false;
        (void) stage;
    }

    // F0 predictor.
    f0_condnet_.resize(5);
    for (int i = 0; i < 5; ++i) {
        char nm[64];
        std::snprintf(nm, sizeof(nm), "mel2wav.f0_predictor.condnet.%d", i * 2);
        ConvSrc cs;
        if (!load_conv_src(nm, cs)) return false;
        all_convs.push_back({std::string("f0.condnet.") + std::to_string(i * 2),
                              cs, &f0_condnet_[i]});
    }
    f0_cls_w_ = get("mel2wav.f0_predictor.classifier.weight");
    f0_cls_b_ = get("mel2wav.f0_predictor.classifier.bias");

    m_src_w_ = get("mel2wav.m_source.l_linear.weight");
    m_src_b_ = get("mel2wav.m_source.l_linear.bias");

    backend_ = chatterbox::default_backend();
    if (!backend_) {
        std::fprintf(stderr, "HiFTVocoder::load: backend init failed\n");
        return false;
    }

    // Stage 2: allocate the derived-weight context.
    // We need:
    //   - one fused weight tensor per ConvSrc (n = all_convs.size())
    //   - two (alpha, alpha_inv) tensors per snake (we point alpha at the
    //     existing model tensor for non-precomputed, but we DO precompute
    //     1/(alpha+eps) into deriv_ctx)
    //
    // Count alphas:
    int n_alphas = 0;
    for (const auto& rb : source_resblocks_) n_alphas += 6;
    for (const auto& rb : resblocks_)         n_alphas += 6;

    const size_t n_tensors = all_convs.size() + n_alphas + 16;
    ggml_init_params p{};
    p.mem_size   = (n_tensors) * ggml_tensor_overhead() + 4096;
    p.no_alloc   = true;
    deriv_ctx_ = ggml_init(p);
    if (!deriv_ctx_) return false;

    // Allocate descriptors for fused convs.
    for (auto& it : all_convs) {
        ConvW* dst = it.dst;
        if (it.src.w_plain) {
            // Plain weight (no fusion needed). We still copy to deriv_ctx
            // to keep all of HiFT's weights in one place (and to keep the
            // dtype unified at fp32 — the plain source_downs weights are
            // fp16 in the GGUF).
            const ggml_tensor* w = it.src.w_plain;
            ggml_tensor* fw = ggml_new_tensor_3d(deriv_ctx_, GGML_TYPE_F16,
                                                    w->ne[0], w->ne[1], w->ne[2]);
            ggml_set_name(fw, ("hift." + it.name + ".w").c_str());
            dst->w = fw;
        } else {
            const ggml_tensor* v = it.src.v;
            ggml_tensor* fw = ggml_new_tensor_3d(deriv_ctx_, GGML_TYPE_F16,
                                                    v->ne[0], v->ne[1], v->ne[2]);
            ggml_set_name(fw, ("hift." + it.name + ".w").c_str());
            dst->w = fw;
        }
        // Bias passthrough (reuse the GGUF tensor).
        dst->b = it.src.bias;
    }

    // Allocate alpha_inv tensors (one per snake instance). Reuse the raw
    // alpha tensor from the GGUF for the non-inverted path.
    auto allocate_inv = [&](HiFTVocoder::ResBlock::Sublayer& S) {
        const int C = static_cast<int>(S.alpha1->ne[0]);
        S.alpha1_inv = ggml_new_tensor_1d(deriv_ctx_, GGML_TYPE_F32, C);
        S.alpha2_inv = ggml_new_tensor_1d(deriv_ctx_, GGML_TYPE_F32, C);
    };
    for (auto& rb : source_resblocks_)
        for (auto& S : rb.sublayers) allocate_inv(S);
    for (auto& rb : resblocks_)
        for (auto& S : rb.sublayers) allocate_inv(S);

    deriv_buffer_ = ggml_backend_alloc_ctx_tensors(deriv_ctx_, backend_);
    if (!deriv_buffer_) {
        std::fprintf(stderr, "HiFTVocoder::load: deriv buffer alloc failed\n");
        return false;
    }

    // Stage 3: fill data.
    auto fp32_to_fp16 = [](const std::vector<float>& in,
                            std::vector<ggml_fp16_t>& out) {
        out.resize(in.size());
        for (size_t i = 0; i < in.size(); ++i) out[i] = ggml_fp32_to_fp16(in[i]);
    };
    std::vector<ggml_fp16_t> tmp_fp16;
    for (auto& it : all_convs) {
        ConvW* dst = it.dst;
        if (it.src.w_plain) {
            // Just copy the plain fp16 (or convert from fp32) bytes.
            std::vector<float> w32 = read_f32(it.src.w_plain);
            fp32_to_fp16(w32, tmp_fp16);
            ggml_backend_tensor_set(dst->w, tmp_fp16.data(), 0,
                                      tmp_fp16.size() * sizeof(ggml_fp16_t));
        } else {
            std::vector<float> fused = fuse_weight_norm_to_numpy(it.src.g, it.src.v);
            fp32_to_fp16(fused, tmp_fp16);
            ggml_backend_tensor_set(dst->w, tmp_fp16.data(), 0,
                                      tmp_fp16.size() * sizeof(ggml_fp16_t));
        }
    }
    auto fill_inv = [&](HiFTVocoder::ResBlock::Sublayer& S) {
        const auto fill = [&](const ggml_tensor* alpha, ggml_tensor* inv) {
            auto a = read_f32(alpha);
            std::vector<float> v(a.size());
            for (size_t i = 0; i < a.size(); ++i)
                v[i] = 1.0f / (a[i] + cfg_.snake_eps);
            ggml_backend_tensor_set(inv, v.data(), 0, v.size() * sizeof(float));
        };
        fill(S.alpha1, S.alpha1_inv);
        fill(S.alpha2, S.alpha2_inv);
    };
    for (auto& rb : source_resblocks_)
        for (auto& S : rb.sublayers) fill_inv(S);
    for (auto& rb : resblocks_)
        for (auto& S : rb.sublayers) fill_inv(S);

    // STFT/iSTFT window: periodic Hann of length n_fft.
    hann_window_.resize(cfg_.n_fft);
    for (int n = 0; n < cfg_.n_fft; ++n) {
        hann_window_[n] = 0.5f - 0.5f *
            std::cos(2.0f * static_cast<float>(M_PI) * n / cfg_.n_fft);
    }
    return true;
}

std::unique_ptr<HiFTVocoder> HiFTVocoder::load(Model* model) {
    if (!model) return nullptr;
    if (model->arch != "chatterbox_s3gen") {
        std::fprintf(stderr,
                     "HiFTVocoder::load: expected chatterbox_s3gen arch, got %s\n",
                     model->arch.c_str());
        return nullptr;
    }
    auto v = std::unique_ptr<HiFTVocoder>(new HiFTVocoder());
    v->model_ = model;
    if (!v->fetch_(model)) return nullptr;
    std::printf("HiFTVocoder::load: ok\n");
    return v;
}


// ----------------------------------------------------------------------------
// STFT / iSTFT (host-side)
// ----------------------------------------------------------------------------

namespace {

// Centered STFT (torch.stft default: pad_mode='reflect', center=True,
// return_complex=True). audio length T -> T_stft = 1 + T // hop frames
// after reflection padding by n_fft//2 each side.
//
// Output real/imag layout: (n_bins, T_stft) row-major.
void stft_centered_host(const std::vector<float>& audio,
                          int n_fft, int hop_len,
                          const std::vector<float>& window,
                          std::vector<float>& real_out,
                          std::vector<float>& imag_out,
                          int& n_bins, int& T_stft) {
    const int T = static_cast<int>(audio.size());
    const int pad = n_fft / 2;
    std::vector<float> padded(T + 2 * pad);
    // Reflect-pad (mirror axis = the edge sample).
    for (int i = 0; i < pad; ++i) padded[pad - 1 - i] = audio[i + 1];
    std::copy(audio.begin(), audio.end(), padded.begin() + pad);
    for (int i = 0; i < pad; ++i) padded[pad + T + i] = audio[T - 2 - i];
    T_stft = 1 + (T + 2 * pad - n_fft) / hop_len;
    n_bins = n_fft / 2 + 1;
    real_out.assign(static_cast<size_t>(n_bins) * T_stft, 0.0f);
    imag_out.assign(static_cast<size_t>(n_bins) * T_stft, 0.0f);
    std::vector<float> frame(n_fft);
    // n_fft is small (16) so a naive DFT is cheap.
    for (int t = 0; t < T_stft; ++t) {
        const float* src = padded.data() + t * hop_len;
        for (int n = 0; n < n_fft; ++n) frame[n] = src[n] * window[n];
        for (int k = 0; k < n_bins; ++k) {
            double sr = 0.0, si = 0.0;
            for (int n = 0; n < n_fft; ++n) {
                const double a = -2.0 * M_PI * k * n / n_fft;
                sr += frame[n] * std::cos(a);
                si += frame[n] * std::sin(a);
            }
            real_out[k * T_stft + t] = static_cast<float>(sr);
            imag_out[k * T_stft + t] = static_cast<float>(si);
        }
    }
}

// Centered iSTFT (torch.istft) -> audio length (T_stft - 1) * hop.
void istft_centered_host(const std::vector<float>& real,
                           const std::vector<float>& imag,
                           int n_fft, int hop_len, int T_stft,
                           const std::vector<float>& window,
                           std::vector<float>& audio_out, int& T_audio) {
    const int n_bins = n_fft / 2 + 1;
    const int audio_len_padded = (T_stft - 1) * hop_len + n_fft;
    std::vector<float> audio_pad(audio_len_padded, 0.0f);
    std::vector<float> weight  (audio_len_padded, 0.0f);
    std::vector<float> frame(n_fft);
    for (int t = 0; t < T_stft; ++t) {
        // IDFT (naive). frame[n] = sum_k (real_k + i imag_k) e^{i 2 pi k n / N}
        // For a real signal we can use symmetry, but n_fft=16 is tiny.
        for (int n = 0; n < n_fft; ++n) {
            double s = 0.0;
            // k=0 contributes real_0 (no symmetry partner; DC term)
            // k in (0, N/2) contributes 2 Re(spec_k e^{i 2pi k n/N})
            // k = N/2 contributes real_(N/2) * (-1)^n (Nyquist; only when N even)
            s += real[0 * T_stft + t];
            for (int k = 1; k < n_bins - 1; ++k) {
                const double a = 2.0 * M_PI * k * n / n_fft;
                s += 2.0 * (real[k * T_stft + t] * std::cos(a)
                              - imag[k * T_stft + t] * std::sin(a));
            }
            // Nyquist bin (only if n_fft is even)
            s += real[(n_bins - 1) * T_stft + t] * ((n & 1) ? -1.0 : 1.0);
            frame[n] = static_cast<float>(s / n_fft);
        }
        for (int n = 0; n < n_fft; ++n) {
            audio_pad[t * hop_len + n] += frame[n] * window[n];
            weight  [t * hop_len + n] += window[n] * window[n];
        }
    }
    for (size_t i = 0; i < audio_pad.size(); ++i) {
        if (weight[i] > 1e-11f) audio_pad[i] /= weight[i];
    }
    const int pad = n_fft / 2;
    T_audio = audio_len_padded - 2 * pad;
    audio_out.assign(audio_pad.begin() + pad,
                       audio_pad.begin() + pad + T_audio);
}

}  // namespace


// ----------------------------------------------------------------------------
// F0 predictor (in-graph)
// ----------------------------------------------------------------------------

std::vector<float> HiFTVocoder::predict_f0(const std::vector<float>& mel,
                                              int T_mel) {
    if (T_mel <= 0 || static_cast<int>(mel.size()) != T_mel * cfg_.in_channels)
        return {};

    const size_t buf_size =
        ggml_tensor_overhead() * 1024
        + ggml_graph_overhead_custom(1024, false);
    std::vector<uint8_t> buf(buf_size);
    ggml_init_params params{};
    params.mem_size   = buf.size();
    params.mem_buffer = buf.data();
    params.no_alloc   = true;
    ggml_context* ctx = ggml_init(params);
    if (!ctx) return {};
    ggml_cgraph* gf = ggml_new_graph_custom(ctx, 1024, false);

    // Input mel (C, T) — numpy (T, C) bytes match.
    ggml_tensor* mel_CT = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,
                                                cfg_.in_channels, T_mel);
    ggml_set_input(mel_CT);

    ggml_tensor* x = mel_CT;
    for (int i = 0; i < 5; ++i) {
        x = conv1d_CT(ctx, x, f0_condnet_[i], 1, 1, 1, cfg_.base_channels);
        x = ggml_elu(ctx, x);
    }
    // x: (512, T). Linear(512 -> 1) via ggml_mul_mat:
    //   weight ne=(in=512, out=1), input ne=(in=512, T) -> out=(1, T).
    ggml_tensor* y = ggml_add(ctx,
        mm_f32(ctx, f0_cls_w_, x), f0_cls_b_);                       // (1, T)
    y = ggml_abs(ctx, y);
    // Reshape (1, T) -> (T,) for the host return.
    y = ggml_reshape_1d(ctx, y, T_mel);
    ggml_set_output(y);
    ggml_build_forward_expand(gf, y);

    ggml_backend_sched_t sched = chatterbox::make_sched(GRAPH_MAX_NODES);
    // Parity is held by PREC_F32 in conv_1d_f32 / mm_f32 — no pin needed.
    if (!sched || !ggml_backend_sched_alloc_graph(sched, gf)) {
        if (sched) ggml_backend_sched_free(sched);
        ggml_free(ctx);
        return {};
    }
    ggml_backend_tensor_set(mel_CT, mel.data(), 0,
                              mel.size() * sizeof(float));
    if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) {
        if (sched) ggml_backend_sched_free(sched);
        ggml_free(ctx);
        return {};
    }
    std::vector<float> out(T_mel);
    ggml_backend_tensor_get(y, out.data(), 0, out.size() * sizeof(float));
    if (sched) ggml_backend_sched_free(sched);
    ggml_free(ctx);
    return out;
}


// ----------------------------------------------------------------------------
// Decode (in-graph, with iSTFT host-side)
// ----------------------------------------------------------------------------

std::vector<float> HiFTVocoder::decode_with_stages(
        const std::vector<float>& mel, int T_mel,
        const std::vector<float>& source, int T_wav,
        std::unordered_map<std::string, std::vector<float>>& stages_out,
        std::unordered_map<std::string, std::pair<int, int>>& stage_shapes) {
    stages_out.clear();
    stage_shapes.clear();

    // ---- 1. Host-side STFT of the source ----
    std::vector<float> s_real_flat, s_imag_flat;
    int n_bins = 0, T_stft = 0;
    stft_centered_host(source, cfg_.n_fft, cfg_.hop_len, hann_window_,
                        s_real_flat, s_imag_flat, n_bins, T_stft);

    // s_stft = concat(real, imag) along channel dim: (18, T_stft).
    // Numpy layout in our reference: (T_stft, 18) row-major, with the
    // first 9 channels = real and last 9 = imag.
    std::vector<float> s_stft_flat(static_cast<size_t>(T_stft) * 2 * n_bins);
    // Build per-row (T, 2*n_bins) layout: row t has [real(0..8, t), imag(0..8, t)].
    for (int t = 0; t < T_stft; ++t) {
        for (int k = 0; k < n_bins; ++k) {
            s_stft_flat[t * (2 * n_bins) + k] = s_real_flat[k * T_stft + t];
            s_stft_flat[t * (2 * n_bins) + n_bins + k] = s_imag_flat[k * T_stft + t];
        }
    }
    {
        // dump source_stft (1, 18, T_stft) — internal stage in the
        // reference. Numpy layout was (1, T_stft, 18).
        stages_out["source_stft"] = s_stft_flat;
        stage_shapes["source_stft"] = {T_stft, 2 * n_bins};
    }

    // ---- 2. Graph: conv_pre -> 3 upsample stages -> conv_post ----
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

    ggml_tensor* mel_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,
                                                cfg_.in_channels, T_mel);
    ggml_tensor* s_stft_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,
                                                  2 * n_bins, T_stft);
    ggml_set_input(mel_in);
    ggml_set_input(s_stft_in);

    std::vector<std::pair<std::string, ggml_tensor*>> stage_tensors;
    auto mark = [&](const char* name, ggml_tensor* tt) {
        ggml_set_name(tt, name);
        ggml_set_output(tt);
        stage_tensors.emplace_back(name, tt);
    };

    // conv_pre: 80 -> 512 (k=7, p=3)
    ggml_tensor* x = conv1d_CT(ctx, mel_in, conv_pre_, 1, 3, 1,
                                  cfg_.base_channels);
    mark("after_conv_pre", x);

    // 3 upsample stages.
    for (int i = 0; i < 3; ++i) {
        const int C_in  = cfg_.base_channels / (1 << i);          // 512, 256, 128
        const int C_out = cfg_.base_channels / (1 << (i + 1));    // 256, 128, 64
        (void) C_in;
        const int s = cfg_.up_strides[i];
        const int k = cfg_.up_kernels[i];
        const int p = (k - s) / 2;

        x = ggml_leaky_relu(ctx, x, cfg_.lrelu_slope, false);
        x = conv_transpose1d_CT(ctx, x, ups_[i], s, p, C_out);
        {
            char nm[32];
            std::snprintf(nm, sizeof(nm), "after_ups%d", i);
            mark(nm, x);
        }

        if (i == 2) {
            // reflection_pad((1, 0)) on the time axis. ne[0]=C, ne[1]=T.
            // Reflect-pad of length 1 on the LEFT (and 0 on the right).
            // Emulate via concat with x[:, 1:2] reversed (just x[:, 1:2]
            // since pad length is 1 and reflect pads with the value at
            // index 1, mirrored). For pad=1 left, we copy x[..., 1].
            const int T_cur = static_cast<int>(x->ne[1]);
            ggml_tensor* col1 = ggml_view_2d(ctx, x,
                                                C_out, 1,
                                                x->nb[1],
                                                /*offset=*/1 * x->nb[1]);
            ggml_tensor* col1_c = ggml_cont(ctx, col1);
            x = ggml_concat(ctx, col1_c, x, /*dim=*/1);    // prepend
            mark("after_refl_pad", x);
            (void) T_cur;
        }

        // source_downs[i]: 18 -> C_out (plain conv, NOT weight-normed)
        int src_pad = (cfg_.src_strides[i] == 1) ? 0 : (cfg_.src_strides[i] / 2);
        ggml_tensor* si = conv1d_CT(ctx, s_stft_in, source_downs_[i],
                                       cfg_.src_strides[i], src_pad, 1, C_out);

        // source_resblocks[i]
        si = resblock_forward(ctx, si, source_resblocks_[i], C_out);
        {
            char nm[32];
            std::snprintf(nm, sizeof(nm), "after_src_rb%d", i);
            mark(nm, si);
        }

        // Add source to x. The T-axis lengths should match.
        x = ggml_add(ctx, x, si);
        {
            char nm[32];
            std::snprintf(nm, sizeof(nm), "after_src_add%d", i);
            mark(nm, x);
        }

        // Parallel ResBlock kernels, then mean.
        ggml_tensor* sum = nullptr;
        for (int j = 0; j < 3; ++j) {
            ggml_tensor* r = resblock_forward(ctx, x, resblocks_[i * 3 + j],
                                                 C_out);
            sum = (sum == nullptr) ? r : ggml_add(ctx, sum, r);
        }
        x = ggml_scale(ctx, sum, 1.0f / 3.0f);
        {
            char nm[32];
            std::snprintf(nm, sizeof(nm), "after_rb_avg%d", i);
            mark(nm, x);
        }
    }

    // conv_post
    x = ggml_leaky_relu(ctx, x, 0.01f, false);
    x = conv1d_CT(ctx, x, conv_post_, 1, 3, 1, 2 * n_bins);
    mark("after_conv_post", x);

    ggml_build_forward_expand(gf, x);

    ggml_backend_sched_t sched = chatterbox::make_sched(GRAPH_MAX_NODES);
    // Parity is held by PREC_F32 in conv_1d_f32 — no pin needed.
    if (!sched || !ggml_backend_sched_alloc_graph(sched, gf)) {
        std::fprintf(stderr, "HiFTVocoder: galloc_alloc_graph failed\n");
        if (sched) ggml_backend_sched_free(sched);
        ggml_free(ctx);
        return {};
    }
    ggml_backend_tensor_set(mel_in, mel.data(), 0, mel.size() * sizeof(float));
    ggml_backend_tensor_set(s_stft_in, s_stft_flat.data(), 0,
                              s_stft_flat.size() * sizeof(float));
    if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "HiFTVocoder: graph_compute failed\n");
        if (sched) ggml_backend_sched_free(sched);
        ggml_free(ctx);
        return {};
    }

    for (const auto& [name, tt] : stage_tensors) {
        const int C_st = static_cast<int>(tt->ne[0]);
        const int T_st = static_cast<int>(tt->ne[1]);
        std::vector<float> out_st(static_cast<size_t>(C_st) * std::max(T_st, 1));
        ggml_backend_tensor_get(tt, out_st.data(), 0,
                                  out_st.size() * sizeof(float));
        stages_out[name] = std::move(out_st);
        stage_shapes[name] = {T_st, C_st};
    }

    // Extract conv_post output and run iSTFT host-side.
    // Layout: ggml ne=(18, T_final). Numpy view (T_final, 18) bytes-equivalent.
    const auto& post_flat = stages_out["after_conv_post"];
    const auto post_shape = stage_shapes["after_conv_post"];
    const int T_final = post_shape.first;
    // magnitude = exp(post[:9, :]), phase = sin(post[9:, :])
    std::vector<float> real(static_cast<size_t>(n_bins) * T_final);
    std::vector<float> imag(static_cast<size_t>(n_bins) * T_final);
    // post_flat is in (T_final, 18) row-major (numpy).
    std::vector<float> mag_dump(static_cast<size_t>(n_bins) * T_final);
    std::vector<float> phase_dump(static_cast<size_t>(n_bins) * T_final);
    for (int t = 0; t < T_final; ++t) {
        for (int k = 0; k < n_bins; ++k) {
            const float mag_raw   = post_flat[t * (2 * n_bins) + k];
            const float phase_raw = post_flat[t * (2 * n_bins) + n_bins + k];
            const float mag   = std::min(std::exp(mag_raw), 100.0f);
            const float phase = std::sin(phase_raw);
            mag_dump  [k * T_final + t] = mag;
            phase_dump[k * T_final + t] = phase;
            real[k * T_final + t] = mag * std::cos(phase);
            imag[k * T_final + t] = mag * std::sin(phase);
        }
    }
    {
        // Dump magnitude / phase for bisect-debug.
        // Layout: (n_bins=9, T_final). To match numpy (1, 9, T_final)
        // dump (-> bytes = T innermost), store as ggml shape (T_final, 9)
        // row-major. Our reference dumps these as `(1, T, C)` so:
        std::vector<float> mag_TC(mag_dump.size());
        std::vector<float> ph_TC(phase_dump.size());
        for (int t = 0; t < T_final; ++t)
            for (int k = 0; k < n_bins; ++k) {
                mag_TC[t * n_bins + k] = mag_dump[k * T_final + t];
                ph_TC [t * n_bins + k] = phase_dump[k * T_final + t];
            }
        stages_out["magnitude"] = mag_TC;
        stage_shapes["magnitude"] = {T_final, n_bins};
        stages_out["phase"]     = ph_TC;
        stage_shapes["phase"]   = {T_final, n_bins};
    }

    std::vector<float> wav;
    int T_audio = 0;
    istft_centered_host(real, imag, cfg_.n_fft, cfg_.hop_len, T_final,
                          hann_window_, wav, T_audio);
    for (auto& v : wav) {
        v = std::clamp(v, -cfg_.audio_limit, cfg_.audio_limit);
    }
    if (T_audio != T_wav) {
        std::fprintf(stderr,
                     "HiFTVocoder::decode: T_audio mismatch (%d vs requested %d)\n",
                     T_audio, T_wav);
    }
    if (sched) ggml_backend_sched_free(sched);
    ggml_free(ctx);
    return wav;
}

std::vector<float> HiFTVocoder::decode(const std::vector<float>& mel, int T_mel,
                                          const std::vector<float>& source,
                                          int T_wav) {
    std::unordered_map<std::string, std::vector<float>> stages;
    std::unordered_map<std::string, std::pair<int, int>> shapes;
    return decode_with_stages(mel, T_mel, source, T_wav, stages, shapes);
}


// ----------------------------------------------------------------------------
// NSF source generator (host-side, matches SineGen + SourceModuleHnNSF)
// ----------------------------------------------------------------------------

std::vector<float> HiFTVocoder::generate_source(
        const std::vector<float>& f0,
        const std::vector<float>& phase_vec_opt,
        const std::vector<float>& noise_z_opt,
        uint64_t                  seed) {
    constexpr int    NB_HARMONICS    = 8;       // 9 total channels
    constexpr float  SINE_AMP        = 0.1f;
    constexpr float  NOISE_STD       = 0.003f;
    constexpr float  VOICED_THR      = 10.0f;
    const int        n_harm          = NB_HARMONICS + 1;
    const int        T_mel           = static_cast<int>(f0.size());
    if (T_mel <= 0) return {};
    const int        upsample        = cfg_.upsample_factor;
    const int        T_wav           = T_mel * upsample;

    // Step 1: nearest-upsample f0 by `upsample` along time. (T_wav,)
    std::vector<float> f0_up(T_wav);
    for (int t = 0; t < T_mel; ++t) {
        const float v = f0[t];
        for (int j = 0; j < upsample; ++j) f0_up[t * upsample + j] = v;
    }

    // Step 2: phase_vec — (n_harm,), uniform[-pi, pi], with [0] = 0.
    std::vector<float> phase_vec(n_harm, 0.0f);
    if (!phase_vec_opt.empty()) {
        if (static_cast<int>(phase_vec_opt.size()) != n_harm) {
            std::fprintf(stderr,
                         "generate_source: phase_vec size %zu != %d\n",
                         phase_vec_opt.size(), n_harm);
            return {};
        }
        phase_vec = phase_vec_opt;
        phase_vec[0] = 0.0f;
    }
    // Otherwise leave zeros — but the upstream draws uniform[-π, π]. For
    // a production C++-only path we draw with std::mt19937. The model is
    // robust to the specific phase / noise realization.
    std::mt19937 rng(seed);
    if (phase_vec_opt.empty()) {
        std::uniform_real_distribution<float> u(-3.14159265358979323846f,
                                                  3.14159265358979323846f);
        for (int h = 0; h < n_harm; ++h) phase_vec[h] = u(rng);
        phase_vec[0] = 0.0f;
    }

    // Step 3: noise_z — ((n_harm) * T_wav,) standard normal.
    std::vector<float> noise_z(static_cast<size_t>(n_harm) * T_wav);
    if (!noise_z_opt.empty()) {
        if (noise_z_opt.size() != noise_z.size()) {
            std::fprintf(stderr,
                         "generate_source: noise_z size %zu != %zu\n",
                         noise_z_opt.size(), noise_z.size());
            return {};
        }
        std::memcpy(noise_z.data(), noise_z_opt.data(),
                     noise_z.size() * sizeof(float));
    } else {
        std::normal_distribution<float> n(0.0f, 1.0f);
        for (auto& v : noise_z) v = n(rng);
    }

    // Step 4: cumulative-sum the instantaneous frequencies for each
    // harmonic and wrap to [0, 1). theta = 2π * (cs - floor(cs)).
    // Then add per-harmonic phase, take sin, scale by SINE_AMP.
    // Combine with U/V mask and noise.
    //
    // Allocate sine_waves as (n_harm, T_wav) row-major. The Linear merge
    // at the end takes (T_wav, n_harm), so we'll just write it that way
    // directly to save a transpose.
    std::vector<float> sine_TH(static_cast<size_t>(T_wav) * n_harm);
    std::vector<float> uv(T_wav);
    // Track per-harmonic cumulative phase to avoid temporary 2D buffer.
    std::vector<double> cs(n_harm, 0.0);
    const double inv_sr = 1.0 / static_cast<double>(cfg_.sampling_rate);
    const double two_pi = 2.0 * 3.14159265358979323846;
    for (int t = 0; t < T_wav; ++t) {
        const float f = f0_up[t];
        const float u = (f > VOICED_THR) ? 1.0f : 0.0f;
        uv[t] = u;
        const float noise_amp = u * NOISE_STD
                                 + (1.0f - u) * (SINE_AMP / 3.0f);
        for (int h = 0; h < n_harm; ++h) {
            const double f_h = static_cast<double>(f) * (h + 1) * inv_sr;
            cs[h] += f_h;
            const double frac = cs[h] - std::floor(cs[h]);
            const double th = two_pi * frac + phase_vec[h];
            const float sine = SINE_AMP * static_cast<float>(std::sin(th));
            const float noise =
                noise_amp * noise_z[static_cast<size_t>(h) * T_wav + t];
            sine_TH[static_cast<size_t>(t) * n_harm + h] =
                sine * u + noise;
        }
    }

    // Step 5: SourceModuleHnNSF.l_linear (9, 1) + bias, then tanh.
    auto lw_vec = read_f32(m_src_w_);                         // (9,)
    auto lb_vec = read_f32(m_src_b_);                         // (1,)
    if (lw_vec.size() != static_cast<size_t>(n_harm)) {
        std::fprintf(stderr, "generate_source: bad l_linear weight\n");
        return {};
    }
    const float lb = lb_vec.empty() ? 0.0f : lb_vec[0];
    std::vector<float> out(T_wav);
    for (int t = 0; t < T_wav; ++t) {
        double s = lb;
        for (int h = 0; h < n_harm; ++h) {
            s += lw_vec[h] *
                  sine_TH[static_cast<size_t>(t) * n_harm + h];
        }
        out[t] = std::tanh(static_cast<float>(s));
    }
    return out;
}

}  // namespace chatterbox
