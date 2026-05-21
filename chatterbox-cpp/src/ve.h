#pragma once

// VoiceEncoder (VE) forward pass.
//
// 3-layer LSTM speaker encoder. Reference: upstream
// chatterbox/models/voice_encoder/voice_encoder.py.
//
//   mel (T, 40)  ->  LSTM(40 -> 256)  ->  LSTM(256 -> 256)  ->  LSTM(256 -> 256)
//   -> last_hidden  ->  Linear(256 -> 256)  ->  ReLU  ->  L2-normalize
//   -> speaker embedding (256,)
//
// Used by T3's cond_enc to turn a reference WAV-derived embedding into
// the speaker conditioning prefix. (This module produces THE 256-d
// speaker_emb that the T3::prefill speaker_emb argument expects.)

#include "model.h"

#include <cstdint>
#include <memory>
#include <vector>

struct ggml_backend;
typedef ggml_backend* ggml_backend_t;

namespace chatterbox {

struct VEConfig {
    int32_t n_mels        = 40;
    int32_t hidden        = 256;
    int32_t n_lstm_layers = 3;
    int32_t emb_dim       = 256;
    bool    final_relu    = true;
};

class VE {
public:
    // Construct from a chatterbox_ve GGUF Model. Returns nullptr on missing
    // tensors / wrong arch.
    static std::unique_ptr<VE> load(Model* model);

    ~VE();
    VE(const VE&) = delete;
    VE& operator=(const VE&) = delete;

    // Run the forward on a (T, n_mels) row-major mel spectrogram.
    // mel_flat must contain exactly n_frames * n_mels floats.
    // Returns the 256-d L2-normalized speaker embedding, or empty on error.
    std::vector<float> forward(const std::vector<float>& mel_flat,
                                int n_frames);

    const VEConfig& config() const { return cfg_; }

private:
    VE() = default;

    VEConfig cfg_;
    Model*   model_ = nullptr;

    ggml_tensor* w_ih_[3]  = {nullptr, nullptr, nullptr};
    ggml_tensor* w_hh_[3]  = {nullptr, nullptr, nullptr};
    ggml_tensor* b_ih_[3]  = {nullptr, nullptr, nullptr};
    ggml_tensor* b_hh_[3]  = {nullptr, nullptr, nullptr};
    ggml_tensor* proj_w_   = nullptr;
    ggml_tensor* proj_b_   = nullptr;

    ggml_backend_t backend_ = nullptr;
};

}  // namespace chatterbox
