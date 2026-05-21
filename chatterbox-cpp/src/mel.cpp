#include "mel.h"

#include "backend.h"

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace chatterbox {

namespace {

// Reflect-pad `x` by `pad` samples on each side. Matches numpy
// mode='reflect' / torch.stft center=True default — edge samples are
// NOT duplicated; the mirror axis sits ONE-PAST the boundary sample.
//
// [a, b, c, d, e] with pad=2  ->  [c, b, a, b, c, d, e, d, c]
void reflect_pad(const std::vector<float>& x, int pad,
                  std::vector<float>& out) {
    const int n = static_cast<int>(x.size());
    out.resize(static_cast<size_t>(n) + 2 * pad);
    for (int i = 0; i < pad; ++i) {
        out[pad - 1 - i] = x[i + 1];
    }
    std::copy(x.begin(), x.end(), out.begin() + pad);
    for (int i = 0; i < pad; ++i) {
        out[pad + n + i] = x[n - 2 - i];
    }
}

// Naive real-input DFT.
}  // namespace

std::unique_ptr<MelExtractor> MelExtractor::load(Model* model) {
    if (!model) return nullptr;
    if (model->arch != "chatterbox_s3gen") {
        std::fprintf(stderr,
                     "MelExtractor::load: expected chatterbox_s3gen arch, got %s\n",
                     model->arch.c_str());
        return nullptr;
    }

    ggml_tensor* filters = model->find_tensor("tokenizer._mel_filters");
    ggml_tensor* window  = model->find_tensor("tokenizer.window");
    if (!filters || !window) {
        std::fprintf(stderr,
                     "MelExtractor::load: missing tokenizer._mel_filters or .window\n");
        return nullptr;
    }
    // Accept fp32 or fp16 — these are host-side preprocessing buffers,
    // not the ggml hot path. We cast on copy.
    auto accepted = [](ggml_type t) {
        return t == GGML_TYPE_F32 || t == GGML_TYPE_F16;
    };
    if (!accepted(filters->type) || !accepted(window->type)) {
        std::fprintf(stderr,
                     "MelExtractor::load: unsupported dtype for filter/window (%d/%d)\n",
                     static_cast<int>(filters->type),
                     static_cast<int>(window->type));
        return nullptr;
    }

    // numpy (N_MELS, N_FFT_BINS) -> ggml ne[0]=N_FFT_BINS, ne[1]=N_MELS.
    if (filters->ne[0] != N_FFT_BINS || filters->ne[1] != N_MELS) {
        std::fprintf(stderr,
                     "MelExtractor::load: mel_filters shape [%lld, %lld], expected [%d, %d]\n",
                     static_cast<long long>(filters->ne[0]),
                     static_cast<long long>(filters->ne[1]),
                     N_FFT_BINS, N_MELS);
        return nullptr;
    }
    if (window->ne[0] != N_FFT) {
        std::fprintf(stderr,
                     "MelExtractor::load: window has %lld elements, expected %d\n",
                     static_cast<long long>(window->ne[0]), N_FFT);
        return nullptr;
    }

    auto m = std::unique_ptr<MelExtractor>(new MelExtractor());
    // chatterbox::read_tensor_f32 handles backend-resident tensors
    // (works for Vulkan-allocated memory; t->data is a device pointer
    // there and unsafe to dereference from the host).
    m->mel_filters_ = chatterbox::read_tensor_f32(filters);
    m->window_      = chatterbox::read_tensor_f32(window);
    if (static_cast<int>(m->mel_filters_.size()) != N_MELS * N_FFT_BINS
        || static_cast<int>(m->window_.size()) != N_FFT) {
        std::fprintf(stderr,
                     "MelExtractor::load: failed to read mel filter/window\n");
        return nullptr;
    }

    // Precompute DFT twiddle tables: row k holds cos/sin(-2*pi*k*n/N_FFT).
    m->dft_cos_.resize(static_cast<size_t>(N_FFT_BINS) * N_FFT);
    m->dft_sin_.resize(static_cast<size_t>(N_FFT_BINS) * N_FFT);
    const double two_pi_over_N = -2.0 * M_PI / static_cast<double>(N_FFT);
    for (int k = 0; k < N_FFT_BINS; ++k) {
        float* cr = m->dft_cos_.data() + static_cast<size_t>(k) * N_FFT;
        float* sr = m->dft_sin_.data() + static_cast<size_t>(k) * N_FFT;
        for (int n = 0; n < N_FFT; ++n) {
            const double a = two_pi_over_N * k * n;
            cr[n] = static_cast<float>(std::cos(a));
            sr[n] = static_cast<float>(std::sin(a));
        }
    }
    return m;
}

std::vector<float> MelExtractor::log_mel(const std::vector<float>& audio,
                                          int& out_n_mels,
                                          int& out_n_frames) const {
    out_n_mels   = N_MELS;
    out_n_frames = 0;
    if (audio.empty()) return {};

    // Reflect-pad to match torch.stft(center=True).
    std::vector<float> padded;
    reflect_pad(audio, N_FFT / 2, padded);

    const int n_frames_raw =
        1 + static_cast<int>((padded.size() - N_FFT) / N_HOP);
    // Upstream drops the trailing pad frame ( [..., :-1] ).
    const int n_frames = n_frames_raw - 1;
    if (n_frames <= 0) return {};
    out_n_frames = n_frames;

    // Per-frame |STFT|² → magnitudes laid out as (n_frames, N_FFT_BINS).
    // DFT as two dot products per bin against the precomputed twiddle
    // tables (no trig in the loop).
    std::vector<float> mag(static_cast<size_t>(n_frames) * N_FFT_BINS);
    std::vector<float> frame(N_FFT);
    for (int t = 0; t < n_frames; ++t) {
        const float* src = padded.data() + static_cast<size_t>(t) * N_HOP;
        for (int n = 0; n < N_FFT; ++n) frame[n] = src[n] * window_[n];
        for (int k = 0; k < N_FFT_BINS; ++k) {
            const float* cr = dft_cos_.data() + static_cast<size_t>(k) * N_FFT;
            const float* si = dft_sin_.data() + static_cast<size_t>(k) * N_FFT;
            double re = 0.0, im = 0.0;
            for (int n = 0; n < N_FFT; ++n) {
                re += frame[n] * cr[n];
                im += frame[n] * si[n];
            }
            mag[t * N_FFT_BINS + k] = static_cast<float>(re * re + im * im);
        }
    }

    // mel = mel_filters (N_MELS, N_FFT_BINS) @ mag.T (N_FFT_BINS, n_frames)
    //     -> (N_MELS, n_frames) row-major.
    std::vector<float> mel(static_cast<size_t>(N_MELS) * n_frames, 0.0f);
    for (int m = 0; m < N_MELS; ++m) {
        const float* fil_row = mel_filters_.data() + m * N_FFT_BINS;
        float* mel_row = mel.data() + static_cast<size_t>(m) * n_frames;
        for (int t = 0; t < n_frames; ++t) {
            const float* mag_row = mag.data() + static_cast<size_t>(t) * N_FFT_BINS;
            double sum = 0.0;
            for (int k = 0; k < N_FFT_BINS; ++k) {
                sum += fil_row[k] * mag_row[k];
            }
            mel_row[t] = static_cast<float>(sum);
        }
    }

    // Log normalization (upstream-exact):
    //   clamp(min=1e-10) -> log10
    //   -> max(., max-8)  (8 dB dynamic range)
    //   -> (+4) / 4
    constexpr float LOG_FLOOR = 1e-10f;
    float max_log = -std::numeric_limits<float>::infinity();
    for (float& v : mel) {
        v = std::log10(std::max(v, LOG_FLOOR));
        if (v > max_log) max_log = v;
    }
    const float floor_v = max_log - 8.0f;
    for (float& v : mel) {
        if (v < floor_v) v = floor_v;
        v = (v + 4.0f) / 4.0f;
    }
    return mel;
}

}  // namespace chatterbox
