#pragma once

// Token sampling for T3 autoregressive generation.
//
// Applies HF-compatible logits processors in the standard order:
//   1. Repetition penalty (Keskar et al. 2019)
//   2. Temperature scaling
//   3. Top-k cutoff
//   4. Top-p (nucleus) — using HF's shift-right convention
//   5. Softmax + multinomial sample
//
// Or, when temperature <= 0, a greedy argmax fast path after stages 1, 3, 4.
//
// The sampler holds its own mt19937 RNG; seed=0 means nondeterministic
// (random_device). Any other seed is fully reproducible.

#include <cstdint>
#include <random>
#include <vector>

namespace chatterbox {

struct SamplingParams {
    float    temperature        = 0.8f;
    int      top_k              = 1000;   // 0 disables (no cap on candidates)
    float    top_p              = 0.95f;  // 1.0 disables
    float    repetition_penalty = 1.2f;   // 1.0 disables
    uint32_t seed               = 0;       // 0 = nondeterministic
};

class Sampler {
public:
    explicit Sampler(const SamplingParams& params);

    // Apply filters in-place to `logits` then sample one token id.
    // `previous_tokens` feeds the repetition penalty step.
    // Returns -1 on empty input.
    int32_t sample(std::vector<float>& logits,
                    const std::vector<int32_t>& previous_tokens);

    const SamplingParams& params() const { return params_; }

private:
    SamplingParams params_;
    std::mt19937   rng_;
};

}  // namespace chatterbox
