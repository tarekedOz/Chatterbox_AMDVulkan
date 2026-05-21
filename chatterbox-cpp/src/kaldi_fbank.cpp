#include "kaldi_fbank.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace chatterbox {

namespace {

// HTK mel scale (natural log; matches kaldi's torchaudio implementation).
//   mel(f) = 1127 * ln(1 + f/700)
float hz_to_mel(float f) {
    return 1127.0f * std::log(1.0f + f / 700.0f);
}

// Cooley-Tukey radix-2 FFT (real input, complex output) at N = 512.
// Returns 257 complex bins. Naive O(N log N).
void rfft_512(const std::vector<float>& in, std::vector<float>& re,
                std::vector<float>& im) {
    constexpr int N = KaldiFbankExtractor::PADDED_SIZE;
    constexpr int N_BINS = KaldiFbankExtractor::N_FFT_BINS;
    re.assign(N_BINS, 0.0f);
    im.assign(N_BINS, 0.0f);

    // Bit-reversed permutation.
    std::vector<float> buf_re(N), buf_im(N);
    int log2N = 9;   // 2^9 = 512
    for (int n = 0; n < N; ++n) {
        int rev = 0;
        for (int b = 0; b < log2N; ++b) {
            if (n & (1 << b)) rev |= 1 << (log2N - 1 - b);
        }
        buf_re[rev] = in[n];
        buf_im[rev] = 0.0f;
    }

    // Iterative Cooley-Tukey.
    for (int s = 1; s <= log2N; ++s) {
        const int m  = 1 << s;
        const int m2 = m >> 1;
        const double angle = -2.0 * M_PI / m;
        const double wr = std::cos(angle);
        const double wi = std::sin(angle);
        for (int k = 0; k < N; k += m) {
            double w_re = 1.0, w_im = 0.0;
            for (int j = 0; j < m2; ++j) {
                const int idx0 = k + j;
                const int idx1 = idx0 + m2;
                const double t_re = w_re * buf_re[idx1] - w_im * buf_im[idx1];
                const double t_im = w_re * buf_im[idx1] + w_im * buf_re[idx1];
                buf_re[idx1] = static_cast<float>(buf_re[idx0] - t_re);
                buf_im[idx1] = static_cast<float>(buf_im[idx0] - t_im);
                buf_re[idx0] = static_cast<float>(buf_re[idx0] + t_re);
                buf_im[idx0] = static_cast<float>(buf_im[idx0] + t_im);
                const double new_wr = w_re * wr - w_im * wi;
                const double new_wi = w_re * wi + w_im * wr;
                w_re = new_wr;
                w_im = new_wi;
            }
        }
    }
    for (int k = 0; k < N_BINS; ++k) {
        re[k] = buf_re[k];
        im[k] = buf_im[k];
    }
}

}  // namespace

std::unique_ptr<KaldiFbankExtractor> KaldiFbankExtractor::create() {
    auto e = std::unique_ptr<KaldiFbankExtractor>(new KaldiFbankExtractor());

    // Povey window: hann(periodic=False, length=FRAME_LENGTH)^0.85
    //   hann[n] = 0.5 - 0.5 cos(2π n / (N-1))
    e->povey_window_.resize(FRAME_LENGTH);
    const float denom = static_cast<float>(FRAME_LENGTH - 1);
    for (int n = 0; n < FRAME_LENGTH; ++n) {
        const float hann = 0.5f - 0.5f *
            std::cos(2.0f * static_cast<float>(M_PI) * n / denom);
        e->povey_window_[n] = std::pow(hann, POVEY_POWER);
    }

    // Mel filterbank (HTK scale, triangular). Matches kaldi's get_mel_banks:
    //   num_fft_bins = padded // 2 = 256   (note: NOT +1)
    //   bin centers in mel are evenly spaced
    //   triangular weights against magnitude^2 spectrum
    //   final filterbank is (80, 256), then zero-padded with one column
    //   to make (80, 257) to align with the 257-bin rfft output.
    const int num_fft_bins = PADDED_SIZE / 2;       // 256
    const float fft_bin_width = static_cast<float>(SAMPLE_RATE) / PADDED_SIZE;
    const float mel_lo = hz_to_mel(LOW_FREQ);
    const float mel_hi = hz_to_mel(HIGH_FREQ);
    const float mel_step = (mel_hi - mel_lo) /
                              static_cast<float>(N_MELS + 1);

    e->mel_filters_.assign(static_cast<size_t>(N_MELS) * N_FFT_BINS, 0.0f);
    for (int b = 0; b < N_MELS; ++b) {
        const float left_mel   = mel_lo + b * mel_step;
        const float center_mel = mel_lo + (b + 1) * mel_step;
        const float right_mel  = mel_lo + (b + 2) * mel_step;
        float* row = e->mel_filters_.data() +
                      static_cast<size_t>(b) * N_FFT_BINS;
        for (int k = 0; k < num_fft_bins; ++k) {
            const float freq_k = fft_bin_width * k;
            const float mel_k  = hz_to_mel(freq_k);
            float w = 0.0f;
            if (mel_k > left_mel && mel_k < right_mel) {
                const float up   = (mel_k - left_mel) / (center_mel - left_mel);
                const float down = (right_mel - mel_k) / (right_mel - center_mel);
                w = std::max(0.0f, std::min(up, down));
            }
            row[k] = w;
        }
        // The Nyquist column (k=256) is always 0 (zero-pad to align
        // with rfft output of 257 bins). Already zero from assign().
    }

    return e;
}

std::vector<float> KaldiFbankExtractor::extract(
        const std::vector<float>& audio,
        int& out_n_frames, int& out_n_mels) const {
    out_n_mels = N_MELS;
    out_n_frames = 0;
    const int n_samp = static_cast<int>(audio.size());
    if (n_samp < FRAME_LENGTH) return {};

    const int n_frames = (n_samp - FRAME_LENGTH) / FRAME_SHIFT + 1;
    out_n_frames = n_frames;

    std::vector<float> out(static_cast<size_t>(n_frames) * N_MELS);

    std::vector<float> frame(PADDED_SIZE, 0.0f);
    std::vector<float> re, im;
    for (int t = 0; t < n_frames; ++t) {
        const int start = t * FRAME_SHIFT;
        // 1. Slice into [0, FRAME_LENGTH); rest is zero.
        std::memcpy(frame.data(), audio.data() + start,
                     FRAME_LENGTH * sizeof(float));
        std::fill(frame.begin() + FRAME_LENGTH, frame.end(), 0.0f);

        // 2. Remove DC offset.
        double sum = 0.0;
        for (int k = 0; k < FRAME_LENGTH; ++k) sum += frame[k];
        const float dc = static_cast<float>(sum / FRAME_LENGTH);
        for (int k = 0; k < FRAME_LENGTH; ++k) frame[k] -= dc;

        // 3. Pre-emphasis: y[0] = x[0] - 0.97*x[0]; y[k] = x[k] - 0.97*x[k-1]
        //    (torchaudio uses 'replicate' padding on the left -> x[-1] = x[0]).
        {
            const float x0 = frame[0];
            float prev = x0;
            for (int k = 0; k < FRAME_LENGTH; ++k) {
                const float cur = frame[k];
                frame[k] = cur - PREEMPH * prev;
                prev = cur;
            }
            (void) x0;
        }

        // 4. Povey window.
        for (int k = 0; k < FRAME_LENGTH; ++k) frame[k] *= povey_window_[k];

        // 5/6. RFFT 512.
        rfft_512(frame, re, im);

        // 7. Power spectrum.
        for (int k = 0; k < N_FFT_BINS; ++k) {
            re[k] = re[k] * re[k] + im[k] * im[k];
        }

        // 8. Mel filterbank multiply -> 80 energies.
        // 9. log(max(energy, LOG_EPS))
        float* out_row = out.data() + static_cast<size_t>(t) * N_MELS;
        for (int b = 0; b < N_MELS; ++b) {
            const float* fil = mel_filters_.data() +
                                static_cast<size_t>(b) * N_FFT_BINS;
            double e = 0.0;
            for (int k = 0; k < N_FFT_BINS; ++k) e += fil[k] * re[k];
            out_row[b] = std::log(std::max(static_cast<float>(e), LOG_EPS));
        }
    }
    return out;
}

}  // namespace chatterbox
