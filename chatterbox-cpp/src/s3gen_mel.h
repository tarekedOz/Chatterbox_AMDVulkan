#pragma once

// S3Gen's mel-spectrogram extractor — 24 kHz / 80-mel / natural-log.
//
// Third mel implementation in the project, distinct from S3Tokenizer's
// (128 mels @ 16 kHz, dynamic-range-clamped log normalization in
// mel.h) and from VE's (pre-baked mel input, 40 mels @ 16 kHz).
//
// Mirrors upstream chatterbox/models/s3gen/utils/mel.py exactly:
//
//     audio (24 kHz mono fp32)
//       -> reflect-pad by (n_fft - hop) / 2 = 720 samples each side
//       -> STFT (n_fft=1920, hop=480, Hann window of size 1920)
//       -> magnitude = sqrt(real^2 + imag^2 + 1e-9)
//       -> mel_filters @ magnitude         (librosa Slaney mel basis)
//       -> log(clamp(min=1e-5))             (dynamic_range_compression)
//
// Mel filterbank is generated in C++ at construction time using the
// Slaney mel scale (piecewise linear below 1000 Hz, logarithmic above)
// with Slaney energy normalization. No external file dependency.
//
// DFT is computed as two precomputed-twiddle matrix products (cos/sin
// tables built once at create()). Same O(N²) arithmetic and accumulation
// order as a naive DFT, but with no trig in the hot loop — runs once at
// conditioning time.

#include <cstdint>
#include <memory>
#include <vector>

namespace chatterbox {

class S3GenMelExtractor {
public:
    static constexpr int   SR          = 24000;
    static constexpr int   N_FFT       = 1920;
    static constexpr int   N_HOP       = 480;       // 50 frames/sec at 24 kHz
    static constexpr int   N_MELS      = 80;
    static constexpr int   N_FFT_BINS  = N_FFT / 2 + 1;  // 961
    static constexpr float FMIN        = 0.0f;
    static constexpr float FMAX        = 8000.0f;
    static constexpr float LOG_CLIP    = 1e-5f;
    static constexpr float MAG_EPS     = 1e-9f;

    static std::unique_ptr<S3GenMelExtractor> create();

    // Compute log-mel of `audio` (24 kHz mono fp32).
    // Output is row-major (n_mels, n_frames). n_frames = floor((n_samples
    // + 2*pad - n_fft) / hop) + 1 where pad = (n_fft - hop) / 2 = 720.
    std::vector<float> log_mel(const std::vector<float>& audio,
                                int& out_n_mels,
                                int& out_n_frames) const;

    // Exposed for the parity test to compare against librosa.
    const std::vector<float>& mel_filterbank() const { return mel_filters_; }
    const std::vector<float>& hann_window()    const { return window_;      }

private:
    S3GenMelExtractor() = default;

    std::vector<float> mel_filters_;  // (N_MELS, N_FFT_BINS) row-major
    std::vector<float> window_;       // (N_FFT,) periodic Hann
    // DFT twiddle tables, (N_FFT_BINS, N_FFT) row-major. Bin k row holds
    // cos/sin(-2*pi*k*n/N_FFT) for n in [0, N_FFT). Precomputed in create()
    // so log_mel's inner loop is a plain dot product.
    std::vector<float> dft_cos_;
    std::vector<float> dft_sin_;
};

}  // namespace chatterbox
