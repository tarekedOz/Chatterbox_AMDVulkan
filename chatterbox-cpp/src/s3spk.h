#pragma once

// CAMPPlus speaker encoder — first piece of the S3Gen port.
//
// Maps Kaldi-style 80-d fbank features (T, 80) to a 192-d speaker
// embedding (NOT L2-normalized — the L2 norm happens later in the flow
// layer). Reference: docs/s3gen-survey.md Component 1, and
// scripts/reference_campplus.py.
//
// Loaded from a chatterbox_s3gen GGUF; tensors live under
// `speaker_encoder.*`. 937 source tensors → 244 derived BN
// (scale, bias) tensors precomputed at load time.

#include "model.h"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

struct ggml_backend;
typedef ggml_backend* ggml_backend_t;
struct ggml_backend_buffer;
typedef ggml_backend_buffer* ggml_backend_buffer_t;
struct ggml_context;

namespace chatterbox {

struct CAMPPlusConfig {
    int32_t feat_dim       = 80;
    int32_t emb_dim        = 192;
    int32_t init_channels  = 128;
    int32_t growth_rate    = 32;
    int32_t bn_size        = 4;        // bn_channels = bn_size * growth = 128
    int32_t seg_len        = 100;      // CAMLayer seg_pooling
    float   bn_eps         = 1e-5f;
    float   stats_eps      = 1e-2f;    // upstream's StatsPool eps for std
};

class S3SpeakerEncoder {
public:
    static std::unique_ptr<S3SpeakerEncoder> load(Model* s3gen_model);

    ~S3SpeakerEncoder();
    S3SpeakerEncoder(const S3SpeakerEncoder&) = delete;
    S3SpeakerEncoder& operator=(const S3SpeakerEncoder&) = delete;

    // Forward: fbank (T, 80) row-major -> 192-d unnormalized embedding.
    std::vector<float> forward(const std::vector<float>& fbank,
                                int T, int n_mels);

    // Debug: run only the FCM head, return its output (numpy view (320, T),
    // ggml layout (T, 320)). Used to bisect parity bugs by comparing
    // against scripts/reference_campplus.py --dump-fcm-bin.
    std::vector<float> forward_fcm_only(const std::vector<float>& fbank,
                                          int T, int n_mels,
                                          int& out_C, int& out_T);

    const CAMPPlusConfig& config() const { return cfg_; }

private:
    S3SpeakerEncoder() = default;

    CAMPPlusConfig cfg_;
    Model*         model_ = nullptr;

    // Raw weight handles (lifetime owned by model_->ctx_), keyed by name
    // with the `speaker_encoder.` prefix stripped.
    std::unordered_map<std::string, ggml_tensor*> weights_;

    // BN scale + bias precomputed at load time, keyed by BN prefix
    // (e.g., "head.bn1", "xvector.block1.tdnnd5.nonlinear1.batchnorm").
    //   scale = w / sqrt(rv + eps)    (or 1/sqrt(rv+eps) when affine=False)
    //   bias  = b - rm * scale          (or -rm * scale       when affine=False)
    std::unordered_map<std::string, ggml_tensor*> bn_scale_;
    std::unordered_map<std::string, ggml_tensor*> bn_bias_;

    // Backing for the derived BN tensors (separate from the model buffer).
    ggml_context*         deriv_ctx_     = nullptr;
    ggml_backend_buffer_t deriv_buffer_  = nullptr;

    ggml_backend_t backend_ = nullptr;

    bool precompute_bn_(const std::string& prefix, bool affine);
};

}  // namespace chatterbox
