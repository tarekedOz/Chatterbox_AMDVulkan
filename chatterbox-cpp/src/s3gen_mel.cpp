#include "s3gen_mel.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace chatterbox {

namespace {

// Slaney mel scale (the librosa default with htk=False).
// Below 1000 Hz the scale is linear; above, logarithmic.
//
// f_sp        = 200 / 3                      (mel per Hz, linear region)
// min_log_hz  = 1000.0
// min_log_mel = min_log_hz / f_sp = 15.0
// logstep     = log(6.4) / 27
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

// Reflect-pad `x` by `pad` samples on each side. Mirror axis sits ONE
// PAST the edge sample, matching numpy mode='reflect' and torch's
// F.pad(... mode='reflect').
void reflect_pad(const std::vector<float>& x, int pad,
                  std::vector<float>& out) {
    const int n = static_cast<int>(x.size());
    out.resize(static_cast<size_t>(n) + 2 * pad);
    for (int i = 0; i < pad; ++i) out[pad - 1 - i] = x[i + 1];
    std::copy(x.begin(), x.end(), out.begin() + pad);
    for (int i = 0; i < pad; ++i) out[pad + n + i] = x[n - 2 - i];
}

void generate_slaney_mel_filterbank(float* weights,
                                     int n_mels, int n_fft_bins,
                                     int sr, int n_fft,
                                     float fmin, float fmax) {
    // Mel anchor frequencies: n_mels + 2 points evenly spaced in mel scale,
    // converted back to Hz. mel_f[i+1] is the i-th filter's center;
    // mel_f[i] and mel_f[i+2] are the band edges.
    std::vector<float> mel_f(n_mels + 2);
    const float mel_min = hz_to_mel_slaney(fmin);
    const float mel_max = hz_to_mel_slaney(fmax);
    for (int i = 0; i < n_mels + 2; ++i) {
        const float m = mel_min + (mel_max - mel_min) *
                                   static_cast<float>(i) /
                                   static_cast<float>(n_mels + 1);
        mel_f[i] = mel_to_hz_slaney(m);
    }

    // FFT bin frequencies: linspace(0, sr/2, n_fft_bins). librosa uses
    // np.linspace which gives endpoint-inclusive — verified against the
    // librosa source. Equivalent to k * sr / n_fft for k in [0, n_fft/2].
    std::vector<float> fft_freqs(n_fft_bins);
    for (int k = 0; k < n_fft_bins; ++k) {
        fft_freqs[k] = static_cast<float>(k) * sr * 0.5f /
                       static_cast<float>(n_fft_bins - 1);
    }

    // Triangular filter weights with Slaney energy normalization.
    for (int i = 0; i < n_mels; ++i) {
        const float lo  = mel_f[i];
        const float mid = mel_f[i + 1];
        const float hi  = mel_f[i + 2];
        const float lo_step = mid - lo;
        const float hi_step = hi - mid;
        const float enorm = 2.0f / (hi - lo);  // Slaney energy norm

        float* row = weights + static_cast<size_t>(i) * n_fft_bins;
        for (int k = 0; k < n_fft_bins; ++k) {
            const float f = fft_freqs[k];
            const float lower = (f - lo)  / lo_step;
            const float upper = (hi - f) / hi_step;
            const float w = std::max(0.0f, std::min(lower, upper));
            row[k] = w * enorm;
        }
    }
}

}  // namespace

std::unique_ptr<S3GenMelExtractor> S3GenMelExtractor::create() {
    auto e = std::unique_ptr<S3GenMelExtractor>(new S3GenMelExtractor());

    // Periodic Hann window of length N_FFT (matches torch.hann_window
    // default periodic=True):  w[n] = 0.5 - 0.5 * cos(2*pi*n/N) for
    // n in [0, N). Note: numpy.hanning is SYMMETRIC; we'd be wrong if
    // we used it.
    e->window_.resize(N_FFT);
    for (int n = 0; n < N_FFT; ++n) {
        e->window_[n] = 0.5f - 0.5f *
            std::cos(2.0f * static_cast<float>(M_PI) *
                     static_cast<float>(n) / static_cast<float>(N_FFT));
    }

    e->mel_filters_.resize(static_cast<size_t>(N_MELS) * N_FFT_BINS);
    generate_slaney_mel_filterbank(e->mel_filters_.data(),
                                    N_MELS, N_FFT_BINS, SR, N_FFT,
                                    FMIN, FMAX);

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

std::vector<float> S3GenMelExtractor::log_mel(const std::vector<float>& audio,
                                               int& out_n_mels,
                                               int& out_n_frames) const {
    out_n_mels = N_MELS;
    out_n_frames = 0;
    if (audio.empty()) return {};

    // Upstream pads by (n_fft - hop) / 2 = 720 samples on each side.
    const int pad = (N_FFT - N_HOP) / 2;
    std::vector<float> padded;
    reflect_pad(audio, pad, padded);

    const int n_frames =
        1 + static_cast<int>((padded.size() - N_FFT) / N_HOP);
    if (n_frames <= 0) return {};
    out_n_frames = n_frames;

    // Per-frame magnitude spectrum: sqrt(|X|^2 + 1e-9). The DFT is two
    // dot products per bin against the precomputed cos/sin tables; same
    // arithmetic as a naive DFT but no trig in the loop.
    std::vector<float> mag(static_cast<size_t>(n_frames) * N_FFT_BINS);
    std::vector<float> frame(N_FFT);
    for (int t = 0; t < n_frames; ++t) {
        const float* src = padded.data() + static_cast<size_t>(t) * N_HOP;
        for (int n = 0; n < N_FFT; ++n) frame[n] = src[n] * window_[n];
        float* row = mag.data() + static_cast<size_t>(t) * N_FFT_BINS;
        for (int k = 0; k < N_FFT_BINS; ++k) {
            const float* cr = dft_cos_.data() + static_cast<size_t>(k) * N_FFT;
            const float* si = dft_sin_.data() + static_cast<size_t>(k) * N_FFT;
            double re = 0.0, im = 0.0;
            for (int n = 0; n < N_FFT; ++n) {
                re += frame[n] * cr[n];
                im += frame[n] * si[n];
            }
            row[k] = std::sqrt(static_cast<float>(re * re + im * im) + MAG_EPS);
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
            double s = 0.0;
            for (int k = 0; k < N_FFT_BINS; ++k) {
                s += fil_row[k] * mag_row[k];
            }
            mel_row[t] = static_cast<float>(s);
        }
    }

    // log(clamp(min=1e-5))
    for (float& v : mel) v = std::log(std::max(v, LOG_CLIP));
    return mel;
}

}  // namespace chatterbox
