#pragma once

// Kaldi-style 80-d log-mel fbank @ 16 kHz. Third mel/fbank extractor
// in the project (distinct from S3Tokenizer's 128-mel @ 16k mel and
// S3Gen's 80-mel @ 24k mel — though all three share the
// pad-window-fft-mel-log scaffold).
//
// Mirrors torchaudio.compliance.kaldi.fbank exactly with the parameters
// the chatterbox CAMPPlus path uses:
//
//   num_mel_bins=80, sample_frequency=16000, dither=0.0
//   (all other params default: frame_length=25ms, frame_shift=10ms,
//    preemphasis=0.97, mel_scale='htk' (1127 * ln(1+f/700)),
//    window_type='povey' (hann^0.85), use_energy=False, use_power=True,
//    use_log_fbank=True, low_freq=20.0, high_freq=8000.0,
//    snip_edges=True, remove_dc_offset=True,
//    round_to_power_of_two=True (= FFT size 512 for 25ms@16k))
//
// Pipeline per frame:
//   1. Slice (window_length = 400 samples).
//   2. Remove DC offset: x -= mean(x).
//   3. Pre-emphasis: y[k] = x[k] - 0.97 * x[k-1];  y[0] = 0.03 * x[0].
//   4. Povey window: w[k] = (0.5 - 0.5*cos(2π k / (N-1)))^0.85.
//   5. Zero-pad to padded_size = 512.
//   6. RFFT 512.
//   7. Power spectrum (real^2 + imag^2), 257 bins, last bin ignored
//      by the filterbank (padded with a zero column).
//   8. HTK mel filterbank (80 bins, triangular, low=20 Hz, high=8000 Hz).
//   9. log(max(energy, FLT_MIN)).

#include <cstdint>
#include <memory>
#include <vector>

namespace chatterbox {

class KaldiFbankExtractor {
public:
    static constexpr int   SAMPLE_RATE       = 16000;
    static constexpr int   FRAME_LENGTH      = 400;     // 25 ms @ 16 kHz
    static constexpr int   FRAME_SHIFT       = 160;     // 10 ms @ 16 kHz
    static constexpr int   PADDED_SIZE       = 512;     // next pow2 of 400
    static constexpr int   N_FFT_BINS        = 257;     // 512/2 + 1
    static constexpr int   N_MELS            = 80;
    static constexpr float LOW_FREQ          = 20.0f;
    static constexpr float HIGH_FREQ         = 8000.0f;
    static constexpr float PREEMPH           = 0.97f;
    static constexpr float POVEY_POWER       = 0.85f;
    static constexpr float LOG_EPS           = 1.1754944e-38f;  // FLT_MIN

    static std::unique_ptr<KaldiFbankExtractor> create();

    // Compute log-mel fbank of `audio` (mono fp32 @ 16 kHz).
    // Returns row-major (n_frames, n_mels) where
    //   n_frames = (n_samples - FRAME_LENGTH) / FRAME_SHIFT + 1
    // (snip_edges=True; no trailing partial-frame).
    std::vector<float> extract(const std::vector<float>& audio,
                                 int& out_n_frames, int& out_n_mels) const;

private:
    KaldiFbankExtractor() = default;

    std::vector<float> povey_window_;          // (FRAME_LENGTH,)
    std::vector<float> mel_filters_;           // (N_MELS, N_FFT_BINS) row-major
                                                // last col always 0
};

}  // namespace chatterbox
