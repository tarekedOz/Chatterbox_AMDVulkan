#include "sampler.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>
#include <random>

namespace chatterbox {

Sampler::Sampler(const SamplingParams& params) : params_(params) {
    if (params_.seed == 0) {
        std::random_device rd;
        rng_.seed(rd());
    } else {
        rng_.seed(params_.seed);
    }
}

namespace {

int32_t argmax(const std::vector<float>& v) {
    int32_t best = 0;
    float   best_l = v[0];
    for (size_t i = 1; i < v.size(); ++i) {
        if (v[i] > best_l) {
            best_l = v[i];
            best   = static_cast<int32_t>(i);
        }
    }
    return best;
}

}  // namespace

int32_t Sampler::sample(std::vector<float>& logits,
                         const std::vector<int32_t>& previous_tokens) {
    if (logits.empty()) return -1;

    // 1. Repetition penalty.
    // For each previously seen token, push its logit toward 0:
    //   logit > 0  ->  logit / penalty
    //   logit <= 0 ->  logit * penalty
    if (params_.repetition_penalty != 1.0f) {
        const float pen = params_.repetition_penalty;
        for (int32_t t : previous_tokens) {
            if (t < 0 || static_cast<size_t>(t) >= logits.size()) continue;
            float& l = logits[t];
            l = (l > 0.0f) ? (l / pen) : (l * pen);
        }
    }

    // 2. Temperature.
    // Greedy fast-path when T <= 0 (rep-penalty has already applied;
    // top_k and top_p haven't, but they would not change the argmax in
    // the greedy regime so it's safe to skip them).
    if (params_.temperature <= 0.0f) {
        return argmax(logits);
    }
    if (params_.temperature != 1.0f) {
        const float inv = 1.0f / params_.temperature;
        for (float& l : logits) l *= inv;
    }

    // 3. Top-k cutoff: keep the k highest logits, mask the rest with -inf.
    if (params_.top_k > 0 && static_cast<size_t>(params_.top_k) < logits.size()) {
        std::vector<float> sorted = logits;
        std::nth_element(sorted.begin(),
                         sorted.begin() + params_.top_k - 1,
                         sorted.end(),
                         std::greater<float>());
        const float threshold = sorted[params_.top_k - 1];
        for (float& l : logits) if (l < threshold) l = -std::numeric_limits<float>::infinity();
    }

    // 4. Top-p (nucleus). Sort by logit desc, compute softmax probs, then
    //    mask any token whose CUMULATIVE-PROBS-BEFORE-IT exceeds top_p.
    //    This is HF's "shift-right" semantics: the first token that crosses
    //    the threshold is kept; everything strictly after it is masked.
    if (params_.top_p > 0.0f && params_.top_p < 1.0f) {
        std::vector<int> idx(logits.size());
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(),
                  [&](int a, int b) { return logits[a] > logits[b]; });

        const float max_l = logits[idx[0]];
        std::vector<float> probs(logits.size(), 0.0f);
        float sum = 0.0f;
        for (int i : idx) {
            const float l = logits[i];
            if (l == -std::numeric_limits<float>::infinity()) break;
            const float p = std::exp(l - max_l);
            probs[i] = p;
            sum += p;
        }
        if (sum > 0.0f) {
            const float inv_sum = 1.0f / sum;
            for (auto& p : probs) p *= inv_sum;

            float cum_prev = 0.0f;
            for (size_t k = 0; k < idx.size(); ++k) {
                const int i = idx[k];
                if (k > 0 && cum_prev > params_.top_p) {
                    logits[i] = -std::numeric_limits<float>::infinity();
                }
                cum_prev += probs[i];
            }
        }
    }

    // 5. Softmax over the remaining logits, multinomial sample.
    float max_l = -std::numeric_limits<float>::infinity();
    for (float l : logits) if (l > max_l) max_l = l;
    if (max_l == -std::numeric_limits<float>::infinity()) {
        // Everything masked out — degenerate. Surface a deterministic
        // fallback rather than UB.
        return 0;
    }

    std::vector<float> probs(logits.size());
    float sum = 0.0f;
    for (size_t i = 0; i < logits.size(); ++i) {
        const float l = logits[i];
        probs[i] = (l == -std::numeric_limits<float>::infinity())
                       ? 0.0f
                       : std::exp(l - max_l);
        sum += probs[i];
    }

    std::uniform_real_distribution<float> dist(0.0f, sum);
    const float r = dist(rng_);
    float cum = 0.0f;
    for (size_t i = 0; i < probs.size(); ++i) {
        cum += probs[i];
        if (cum >= r) return static_cast<int32_t>(i);
    }
    return static_cast<int32_t>(probs.size() - 1);
}

}  // namespace chatterbox
