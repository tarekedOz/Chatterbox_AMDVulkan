#pragma once

// Log-mel spectrogram extraction for the S3 audio tokenizer.
//
// Pure host-side computation — no ggml dependency. This runs once per
// reference WAV at conditioning time. The DFT uses precomputed cos/sin
// twiddle tables (built in load()) so the per-frame loop is a plain dot
// product instead of recomputing trig.
//
// Mirrors upstream S3Tokenizer.log_mel_spectrogram exactly:
//   stft(audio, n_fft=400, hop=160, window=hann_400, return_complex=True)
//   mag = stft[..., :-1].abs() ** 2          (drop trailing pad frame)
//   mel = mel_filters @ mag                   (128, 201) @ (201, T)
//   log = clamp(mel, min=1e-10).log10()
//   log = max(log, log.max() - 8.0)           (8-dB dynamic range)
//   log = (log + 4.0) / 4.0                   (normalize to ~[-1, +1])
//
// The Hann window and the mel filterbank are loaded from the s3gen GGUF
// (registered as buffers in upstream, stored as ordinary fp32 tensors
// in our converted GGUF).

#include "model.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace chatterbox {

class MelExtractor {
public:
    static constexpr int SR         = 16000;
    static constexpr int N_FFT      = 400;
    static constexpr int N_HOP      = 160;
    static constexpr int N_MELS     = 128;
    static constexpr int N_FFT_BINS = N_FFT / 2 + 1;     // 201

    // Construct from a chatterbox_s3gen GGUF Model. The filterbank and
    // Hann window are copied out of the GGUF once; the model can be
    // dropped afterwards if desired.
    static std::unique_ptr<MelExtractor> load(Model* s3gen_model);

    // Compute log-mel of `audio` (16 kHz mono fp32 samples).
    // Output is row-major (n_mels, n_frames) — element [m, t] at
    // offset m * n_frames + t. n_frames matches the upstream STFT
    // (center=True + trailing-frame drop): n_frames = audio_len / 160.
    std::vector<float> log_mel(const std::vector<float>& audio,
                                int& out_n_mels,
                                int& out_n_frames) const;

private:
    MelExtractor() = default;

    std::vector<float> mel_filters_;  // (N_MELS, N_FFT_BINS) row-major
    std::vector<float> window_;       // (N_FFT,)
    // DFT twiddle tables (N_FFT_BINS, N_FFT) row-major, built in load().
    std::vector<float> dft_cos_;
    std::vector<float> dft_sin_;
};

}  // namespace chatterbox
