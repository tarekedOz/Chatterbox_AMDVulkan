#pragma once

// S3 audio tokenizer: full WAV -> discrete speech token id pipeline.
//
// Composes the three stages already implemented in chatterbox-cpp:
//   1. MelExtractor   (mel.h)      — host-side STFT + filterbank + log norm.
//   2. S3Encoder      (s3enc.h)    — ggml graph for AudioEncoderV2.
//   3. FSQ quantizer  (host-side)  — Linear(1280 -> 8) + tanh + ×0.999 +
//                                    round + base-3 encode -> [0, 6561).
//
// Reference (NumPy): scripts/reference_s3_tokenizer.py.
// Survey:            docs/s3-tokenizer-survey.md.
//
// One token per 40 ms of audio (25 tokens/sec). The audio is padded to
// a multiple of 640 samples first, matching upstream's
// S3Tokenizer.pad() behaviour.

#include "mel.h"
#include "model.h"
#include "s3enc.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace chatterbox {

class S3Tokenizer {
public:
    static constexpr int SR              = 16000;
    static constexpr int TOKEN_HOP       = 640;        // samples per token
    static constexpr int FSQ_DIM         = 8;
    static constexpr int FSQ_LEVEL       = 3;          // levels per dim
    static constexpr int VOCAB_SIZE      = 6561;        // 3^8 (SPEECH_VOCAB_SIZE)
    static constexpr float FSQ_SCALE     = 0.9990000128746033f;

    static std::unique_ptr<S3Tokenizer> load(Model* s3gen_model);

    ~S3Tokenizer() = default;
    S3Tokenizer(const S3Tokenizer&) = delete;
    S3Tokenizer& operator=(const S3Tokenizer&) = delete;

    // Encode a 16 kHz mono fp32 waveform to speech token ids in [0, 6561).
    // Returns one token per 40 ms of audio (after rounding up to a
    // 640-sample multiple, matching upstream pad()).
    std::vector<int32_t> encode(const std::vector<float>& audio);

private:
    S3Tokenizer() = default;

    std::unique_ptr<MelExtractor> mel_;
    std::unique_ptr<S3Encoder>    enc_;

    // FSQ project_down (Linear 1280 -> 8). Owned copies (fp32). Layout:
    //   proj_w_[j * n_state + k] = w[j, k]   (numpy row-major)
    std::vector<float> proj_w_;
    std::vector<float> proj_b_;
    int n_state_ = 1280;
};

}  // namespace chatterbox
