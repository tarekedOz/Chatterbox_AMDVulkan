// Deterministic unit tests for chatterbox::Sampler.
//
// Each test constructs known logits, runs the sampler with carefully
// chosen params, and asserts the resulting token (or filtered logits)
// against a hand-derived expected value.

#include "sampler.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

namespace {

bool is_neg_inf(float x) {
    return x == -std::numeric_limits<float>::infinity();
}

int failures = 0;

#define CHECK(cond, msg)                                                    \
    do {                                                                    \
        if (!(cond)) {                                                      \
            std::printf("FAIL: %s  (%s)\n", msg, #cond);                    \
            ++failures;                                                     \
        }                                                                   \
    } while (0)

}  // namespace

int main() {
    using chatterbox::Sampler;
    using chatterbox::SamplingParams;

    // -------------------------------------------------------------
    // Test 1: greedy (T=0) returns argmax.
    // -------------------------------------------------------------
    {
        SamplingParams p;
        p.temperature = 0.0f;
        p.top_k = 0;
        p.top_p = 1.0f;
        p.repetition_penalty = 1.0f;
        p.seed = 42;
        Sampler s(p);
        std::vector<float> logits = {1.0f, 3.0f, 2.0f};
        const int32_t t = s.sample(logits, {});
        CHECK(t == 1, "Test 1 (greedy argmax)");
    }

    // -------------------------------------------------------------
    // Test 2: repetition penalty + greedy.
    // logits=[1,3,2], previous=[1] (token id 1 was generated).
    // With penalty=2.0 and logits[1]>0: logits[1] /= 2 = 1.5.
    // New logits = [1.0, 1.5, 2.0]. Argmax = 2.
    // -------------------------------------------------------------
    {
        SamplingParams p;
        p.temperature = 0.0f;
        p.top_k = 0;
        p.top_p = 1.0f;
        p.repetition_penalty = 2.0f;
        p.seed = 42;
        Sampler s(p);
        std::vector<float> logits = {1.0f, 3.0f, 2.0f};
        const int32_t t = s.sample(logits, {1});
        CHECK(t == 2, "Test 2 (rep penalty pushes argmax)");
    }

    // -------------------------------------------------------------
    // Test 3: negative-logit branch of repetition penalty.
    // logits=[-1, -2, -3], previous=[0] (id 0 was generated).
    // logits[0] = -1 <= 0, so logits[0] *= penalty=2 -> -2.
    // New: [-2, -2, -3]. Argmax breaks tie at first index (we take
    // strict greater for argmax). So argmax is still 0.
    // To make it less ambiguous use distinct values:
    // logits=[-1, -3, -2], previous=[0]: logits[0] = -2.
    // New: [-2, -3, -2]. argmax → 0 (tied) or 2 (depending on impl).
    // Our impl takes strict-greater, so first-encountered max stays.
    // -------------------------------------------------------------
    {
        SamplingParams p;
        p.temperature = 0.0f;
        p.repetition_penalty = 2.0f;
        p.top_k = 0;
        p.top_p = 1.0f;
        p.seed = 42;
        Sampler s(p);
        std::vector<float> logits = {-1.0f, -5.0f, -2.0f};
        const int32_t t = s.sample(logits, {0});
        // After penalty: logits[0] = -2. New = [-2, -5, -2]. Strict-> first.
        CHECK(t == 0, "Test 3 (negative-branch rep penalty)");
    }

    // -------------------------------------------------------------
    // Test 4: top_k=2 with greedy.
    // logits=[1,3,2,0.5]. Top-2 = {1, 2} with values {3, 2}.
    // logits after: [-inf, 3, 2, -inf]. Argmax = 1.
    // -------------------------------------------------------------
    {
        SamplingParams p;
        p.temperature = 1.0f;        // non-greedy to verify masking happens
        p.top_k = 2;
        p.top_p = 1.0f;
        p.repetition_penalty = 1.0f;
        p.seed = 42;
        Sampler s(p);
        std::vector<float> logits = {1.0f, 3.0f, 2.0f, 0.5f};
        const int32_t t = s.sample(logits, {});
        CHECK(t == 1 || t == 2, "Test 4 (top_k returns top-k member)");
        CHECK(is_neg_inf(logits[0]) && is_neg_inf(logits[3]),
              "Test 4 (top_k masks below-threshold)");
    }

    // -------------------------------------------------------------
    // Test 5: top_p hands the strongest token, drops the weakest.
    // Construct logits so probabilities are dominated by token 0.
    // logits=[10, 5, 3, 2]. Softmax: ~[~1, ~0.007, ~0.001, ~0.0004].
    // top_p=0.5 -> keep only token 0 (cum_prev > 0.5 is true for k>=1).
    // Greedy on remaining → 0.
    // -------------------------------------------------------------
    {
        SamplingParams p;
        p.temperature = 1.0f;
        p.top_k = 0;
        p.top_p = 0.5f;
        p.repetition_penalty = 1.0f;
        p.seed = 42;
        Sampler s(p);
        std::vector<float> logits = {10.0f, 5.0f, 3.0f, 2.0f};
        const int32_t t = s.sample(logits, {});
        CHECK(t == 0, "Test 5 (top_p narrows to dominant token)");
        CHECK(is_neg_inf(logits[1]) && is_neg_inf(logits[2]) && is_neg_inf(logits[3]),
              "Test 5 (top_p masks the tail)");
    }

    // -------------------------------------------------------------
    // Test 6: determinism across two instances with the same seed.
    // -------------------------------------------------------------
    {
        SamplingParams p;
        p.temperature = 1.0f;
        p.top_k = 0;
        p.top_p = 1.0f;
        p.repetition_penalty = 1.0f;
        p.seed = 123;
        Sampler a(p);
        Sampler b(p);
        std::vector<float> la = {1.0f, 2.0f, 3.0f, 2.0f, 1.0f};
        std::vector<float> lb = la;
        const int32_t ta = a.sample(la, {});
        const int32_t tb = b.sample(lb, {});
        CHECK(ta == tb, "Test 6 (same seed same draw)");
    }

    // -------------------------------------------------------------
    // Test 7: very high temperature flattens the distribution. With
    // 5 candidates and 1000 samples, each bucket should land near 200.
    // -------------------------------------------------------------
    {
        SamplingParams p;
        p.temperature = 1000.0f;    // T -> infinity ≈ uniform
        p.top_k = 0;
        p.top_p = 1.0f;
        p.repetition_penalty = 1.0f;
        p.seed = 42;
        Sampler s(p);
        std::vector<int> counts(5, 0);
        for (int i = 0; i < 1000; ++i) {
            std::vector<float> logits = {1.0f, 2.0f, 3.0f, 2.0f, 1.0f};
            const int32_t t = s.sample(logits, {});
            ++counts[t];
        }
        bool ok = true;
        for (int c : counts) if (c < 130 || c > 270) ok = false;
        if (!ok) {
            std::printf("Test 7 (uniform under high T) counts: [%d %d %d %d %d]\n",
                        counts[0], counts[1], counts[2], counts[3], counts[4]);
        }
        CHECK(ok, "Test 7 (uniform under high T)");
    }

    if (failures == 0) {
        std::printf("All sampler tests passed.\n");
        return 0;
    }
    std::printf("%d sampler tests failed.\n", failures);
    return 1;
}
