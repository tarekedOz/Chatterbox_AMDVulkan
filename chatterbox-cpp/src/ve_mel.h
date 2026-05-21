#pragma once

// VE 40-mel @ 16 kHz extractor. Fourth mel-family extractor in the
// project; produces the input to the VoiceEncoder LSTM.
//
// Mirrors chatterbox/models/voice_encoder/melspec.py exactly:
//
//   audio (mono 16 kHz fp32)
//     -> librosa.stft(n_fft=400, hop=160, win=400, window=hann periodic,
//                     center=True, pad_mode='reflect')
//     -> power spectrum (real^2 + imag^2)
//     -> librosa-Slaney mel filterbank (40 mels, fmin=0, fmax=8000)
//     -> log(clamp(min=1e-5))
//
// Output: row-major (T, 40) — same byte layout as VE expects.

#include <cstdint>
#include <memory>
#include <vector>

namespace chatterbox {

class VEMelExtractor {
public:
    static constexpr int   SAMPLE_RATE   = 16000;
    static constexpr int   N_FFT         = 400;
    static constexpr int   HOP_LEN       = 160;
    static constexpr int   N_MELS        = 40;
    static constexpr int   N_FFT_BINS    = N_FFT / 2 + 1;  // 201
    static constexpr float FMIN          = 0.0f;
    static constexpr float FMAX          = 8000.0f;
    static constexpr float LOG_CLIP      = 1e-5f;

    static std::unique_ptr<VEMelExtractor> create();

    std::vector<float> compute(const std::vector<float>& audio,
                                  int& out_n_frames, int& out_n_mels) const;

private:
    VEMelExtractor() = default;

    std::vector<float> mel_filters_;     // (N_MELS, N_FFT_BINS) row-major
    std::vector<float> window_;          // (N_FFT,) periodic Hann
    // DFT twiddle tables (N_FFT_BINS, N_FFT) row-major, built in create()
    // so compute() avoids per-sample trig.
    std::vector<float> dft_cos_;
    std::vector<float> dft_sin_;
};

}  // namespace chatterbox
