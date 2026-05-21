#include "s3tok.h"

#include "backend.h"

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace chatterbox {

std::unique_ptr<S3Tokenizer> S3Tokenizer::load(Model* model) {
    if (!model) return nullptr;

    auto mel = MelExtractor::load(model);
    auto enc = S3Encoder::load(model);
    if (!mel || !enc) {
        std::fprintf(stderr, "S3Tokenizer::load: mel or encoder load failed\n");
        return nullptr;
    }

    ggml_tensor* pw = model->find_tensor("tokenizer.quantizer._codebook.project_down.weight");
    ggml_tensor* pb = model->find_tensor("tokenizer.quantizer._codebook.project_down.bias");
    if (!pw || !pb) {
        std::fprintf(stderr, "S3Tokenizer::load: missing FSQ project_down tensors\n");
        return nullptr;
    }

    auto tok = std::unique_ptr<S3Tokenizer>(new S3Tokenizer());
    tok->n_state_ = enc->config().n_state;
    tok->mel_ = std::move(mel);
    tok->enc_ = std::move(enc);

    // Backend-aware reads: pw/pb may live in Vulkan memory.
    const size_t n_w = static_cast<size_t>(FSQ_DIM) * tok->n_state_;
    tok->proj_w_ = read_tensor_f32(pw);
    tok->proj_b_ = read_tensor_f32(pb);
    if (tok->proj_w_.size() != n_w
        || static_cast<int>(tok->proj_b_.size()) != FSQ_DIM) {
        std::fprintf(stderr,
                     "S3Tokenizer::load: project_down read failed "
                     "(got %zu / %zu)\n",
                     tok->proj_w_.size(), tok->proj_b_.size());
        return nullptr;
    }
    return tok;
}

std::vector<int32_t> S3Tokenizer::encode(const std::vector<float>& audio) {
    if (audio.empty()) return {};

    // Pad to a multiple of TOKEN_HOP (= 640 samples = 40 ms at 16 kHz).
    // Matches upstream S3Tokenizer.pad().
    const int n_tokens_pad =
        static_cast<int>((audio.size() + TOKEN_HOP - 1) / TOKEN_HOP);
    const int padded_len = n_tokens_pad * TOKEN_HOP;
    std::vector<float> padded(padded_len, 0.0f);
    std::memcpy(padded.data(), audio.data(), audio.size() * sizeof(float));

    int n_mels = 0, T_mel = 0;
    auto log_mel = mel_->log_mel(padded, n_mels, T_mel);
    if (log_mel.empty()) return {};

    int T_tok = 0, n_state = 0;
    auto hidden = enc_->forward(log_mel, n_mels, T_mel, T_tok, n_state);
    if (hidden.empty()) return {};

    // FSQ encode.
    //   h = hidden_t @ proj_w.T + proj_b               (T_tok, 8)
    //   h = tanh(h) * 0.999
    //   q = round(h) + 1                                in {0, 1, 2}^8
    //   token = sum q[i] * 3^i
    static constexpr int32_t POWERS[FSQ_DIM] = {
        1, 3, 9, 27, 81, 243, 729, 2187,
    };

    std::vector<int32_t> tokens(static_cast<size_t>(T_tok));
    for (int t = 0; t < T_tok; ++t) {
        const float* row = hidden.data() + static_cast<size_t>(t) * n_state;
        float h[FSQ_DIM];
        for (int j = 0; j < FSQ_DIM; ++j) {
            double s = 0.0;
            const float* w_row = proj_w_.data() + static_cast<size_t>(j) * n_state;
            for (int k = 0; k < n_state; ++k) {
                s += static_cast<double>(w_row[k]) * row[k];
            }
            h[j] = std::tanh(static_cast<float>(s) + proj_b_[j]) * FSQ_SCALE;
        }
        int32_t token = 0;
        for (int j = 0; j < FSQ_DIM; ++j) {
            int q = static_cast<int>(std::lround(h[j])) + 1;  // {-1,0,+1} -> {0,1,2}
            if (q < 0) q = 0;
            if (q > 2) q = 2;
            token += q * POWERS[j];
        }
        tokens[t] = token;
    }
    return tokens;
}

}  // namespace chatterbox
