#include "ve_mel.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace chatterbox {

namespace {

float hz_to_mel_slaney(float f) {
    constexpr float f_sp = 200.0f / 3.0f;
    if (f < 1000.0f) return f / f_sp;
    const float logstep = std::log(6.4f) / 27.0f;
    return 15.0f + std::log(f / 1000.0f) / logstep;
}

float mel_to_hz_slaney(float m) {
    constexpr float f_sp = 200.0f / 3.0f;
    if (m < 15.0f) return m * f_sp;
    const float logstep = std::log(6.4f) / 27.0f;
    return 1000.0f * std::exp((m - 15.0f) * logstep);
}

void reflect_pad(const std::vector<float>& x, int pad,
                  std::vector<float>& out) {
    const int n = static_cast<int>(x.size());
    out.resize(static_cast<size_t>(n) + 2 * pad);
    for (int i = 0; i < pad; ++i) out[pad - 1 - i] = x[i + 1];
    std::copy(x.begin(), x.end(), out.begin() + pad);
    for (int i = 0; i < pad; ++i) out[pad + n + i] = x[n - 2 - i];
}

void generate_slaney_mel_filterbank(float* w, int n_mels, int n_fft_bins,
                                       int sr, int n_fft,
                                       float fmin, float fmax) {
    std::vector<float> mel_f(n_mels + 2);
    const float mel_min = hz_to_mel_slaney(fmin);
    const float mel_max = hz_to_mel_slaney(fmax);
    for (int i = 0; i < n_mels + 2; ++i) {
        const float m = mel_min + (mel_max - mel_min) *
                                   static_cast<float>(i) /
                                   static_cast<float>(n_mels + 1);
        mel_f[i] = mel_to_hz_slaney(m);
    }
    std::vector<float> fft_freqs(n_fft_bins);
    for (int k = 0; k < n_fft_bins; ++k) {
        fft_freqs[k] = static_cast<float>(k) * sr * 0.5f /
                       static_cast<float>(n_fft_bins - 1);
    }
    for (int i = 0; i < n_mels; ++i) {
        const float lo = mel_f[i], mid = mel_f[i + 1], hi = mel_f[i + 2];
        const float lo_step = mid - lo;
        const float hi_step = hi - mid;
        const float enorm = 2.0f / (hi - lo);
        float* row = w + static_cast<size_t>(i) * n_fft_bins;
        for (int k = 0; k < n_fft_bins; ++k) {
            const float f = fft_freqs[k];
            const float lower = (f - lo)  / lo_step;
            const float upper = (hi - f) / hi_step;
            const float ww = std::max(0.0f, std::min(lower, upper));
            row[k] = ww * enorm;
        }
    }
}

}  // namespace

std::unique_ptr<VEMelExtractor> VEMelExtractor::create() {
    auto e = std::unique_ptr<VEMelExtractor>(new VEMelExtractor());
    // Periodic Hann window of length N_FFT.
    e->window_.resize(N_FFT);
    for (int n = 0; n < N_FFT; ++n) {
        e->window_[n] = 0.5f - 0.5f *
            std::cos(2.0f * static_cast<float>(M_PI) * n / N_FFT);
    }
    e->mel_filters_.resize(static_cast<size_t>(N_MELS) * N_FFT_BINS);
    generate_slaney_mel_filterbank(e->mel_filters_.data(),
                                     N_MELS, N_FFT_BINS, SAMPLE_RATE,
                                     N_FFT, FMIN, FMAX);

    // Precompute DFT twiddle tables: row k holds cos/sin(-2*pi*k*n/N_FFT).
    e->dft_cos_.resize(static_cast<size_t>(N_FFT_BINS) * N_FFT);
    e->dft_sin_.resize(static_cast<size_t>(N_FFT_BINS) * N_FFT);
    const double two_pi_over_N = -2.0 * M_PI / static_cast<double>(N_FFT);
    for (int k = 0; k < N_FFT_BINS; ++k) {
        float* cr = e->dft_cos_.data() + static_cast<size_t>(k) * N_FFT;
        float* sr = e->dft_sin_.data() + static_cast<size_t>(k) * N_FFT;
        for (int n = 0; n < N_FFT; ++n) {
            const double a = two_pi_over_N * k * n;
            cr[n] = static_cast<float>(std::cos(a));
            sr[n] = static_cast<float>(std::sin(a));
        }
    }
    return e;
}

std::vector<float> VEMelExtractor::compute(
        const std::vector<float>& audio,
        int& out_n_frames, int& out_n_mels) const {
    out_n_mels = N_MELS;
    out_n_frames = 0;
    if (audio.empty()) return {};

    // Centered STFT: reflect-pad by N_FFT/2 on each side.
    const int pad = N_FFT / 2;
    std::vector<float> padded;
    reflect_pad(audio, pad, padded);
    const int n_frames =
        1 + static_cast<int>((padded.size() - N_FFT) / HOP_LEN);
    if (n_frames <= 0) return {};
    out_n_frames = n_frames;

    // Per-frame power spectrum. DFT as two dot products per bin against
    // the precomputed twiddle tables (no trig in the loop).
    std::vector<float> frame(N_FFT);
    std::vector<float> power(static_cast<size_t>(n_frames) * N_FFT_BINS);
    for (int t = 0; t < n_frames; ++t) {
        const float* src = padded.data() + static_cast<size_t>(t) * HOP_LEN;
        for (int n = 0; n < N_FFT; ++n) frame[n] = src[n] * window_[n];
        float* row = power.data() + static_cast<size_t>(t) * N_FFT_BINS;
        for (int k = 0; k < N_FFT_BINS; ++k) {
            const float* cr = dft_cos_.data() + static_cast<size_t>(k) * N_FFT;
            const float* si = dft_sin_.data() + static_cast<size_t>(k) * N_FFT;
            double re = 0.0, im = 0.0;
            for (int n = 0; n < N_FFT; ++n) {
                re += frame[n] * cr[n];
                im += frame[n] * si[n];
            }
            row[k] = static_cast<float>(re * re + im * im);
        }
    }

    // Mel filterbank multiply -> (T, N_MELS) row-major directly.
    std::vector<float> out(static_cast<size_t>(n_frames) * N_MELS);
    for (int t = 0; t < n_frames; ++t) {
        const float* prow = power.data() + static_cast<size_t>(t) * N_FFT_BINS;
        float* orow = out.data() + static_cast<size_t>(t) * N_MELS;
        for (int m = 0; m < N_MELS; ++m) {
            const float* fil = mel_filters_.data() +
                                static_cast<size_t>(m) * N_FFT_BINS;
            double s = 0.0;
            for (int k = 0; k < N_FFT_BINS; ++k) s += fil[k] * prow[k];
            orow[m] = std::log(std::max(static_cast<float>(s), LOG_CLIP));
        }
    }
    return out;
}

}  // namespace chatterbox
