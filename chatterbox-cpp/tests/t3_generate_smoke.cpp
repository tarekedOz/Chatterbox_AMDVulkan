// Smoke test for T3 autoregressive speech-token generation.
//
// Reuses the seeded speaker_emb + text tokens from tests/t3_reference.bin
// (produced by scripts/reference_t3_forward.py). Prefills, runs the AR
// loop with a fixed sampler seed, and asserts the result is plausible:
//
//   - generate returns a non-empty vector OR terminates within
//     `max_tokens` with at-or-above-vocab-limit logits (we accept
//     either outcome as long as the loop terminated cleanly).
//   - every token id is in [0, 6561).
//   - n_past_ stayed below max_context.
//
// This is NOT a parity test — sampling adds RNG, and we don't have a
// torch-side oracle for the AR loop. Validation is "runs, terminates,
// emits valid token ids."

#include "model.h"
#include "sampler.h"
#include "t3.h"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "Usage: %s <t3.gguf> <t3_reference.bin>\n",
                     argv[0]);
        return 2;
    }

    auto m = chatterbox::load_model(argv[1]);
    if (!m) return 1;

    auto t3 = chatterbox::T3::load(m.get(), /*max_context=*/512);
    if (!t3) {
        std::fprintf(stderr, "Failed to construct T3.\n");
        return 1;
    }

    std::ifstream in(argv[2], std::ios::binary);
    if (!in) {
        std::fprintf(stderr, "Cannot open reference: %s\n", argv[2]);
        return 1;
    }
    int32_t n_tokens = 0;
    in.read(reinterpret_cast<char*>(&n_tokens), sizeof(n_tokens));
    std::vector<int32_t> tokens(n_tokens);
    in.read(reinterpret_cast<char*>(tokens.data()), n_tokens * sizeof(int32_t));
    int32_t spk_dim = 0;
    in.read(reinterpret_cast<char*>(&spk_dim), sizeof(spk_dim));
    std::vector<float> speaker_emb(spk_dim);
    in.read(reinterpret_cast<char*>(speaker_emb.data()),
            spk_dim * sizeof(float));
    int32_t n_prompt = 0;
    in.read(reinterpret_cast<char*>(&n_prompt), sizeof(n_prompt));
    std::vector<int32_t> prompt(n_prompt);
    in.read(reinterpret_cast<char*>(prompt.data()),
            n_prompt * sizeof(int32_t));
    if (!in) {
        std::fprintf(stderr, "Reference file too short.\n");
        return 1;
    }

    std::printf("Prefill: %d text tokens + %d-d speaker_emb + %d prompt speech tokens\n",
                n_tokens, spk_dim, n_prompt);
    auto pre = t3->prefill(speaker_emb, prompt, tokens);
    if (pre.empty()) {
        std::fprintf(stderr, "prefill returned empty.\n");
        return 1;
    }
    std::printf("After prefill: n_past = %d\n", t3->n_past());

    chatterbox::T3::GenParams params;
    params.sampling.seed = 42;
    params.sampling.temperature = 0.8f;
    params.sampling.top_k = 50;
    params.sampling.top_p = 0.95f;
    params.sampling.repetition_penalty = 1.2f;
    params.max_tokens = 20;
    params.start_token = 6561;
    params.vocab_limit = 6561;

    auto speech = t3->generate(params);

    std::printf("Generated %zu speech tokens, n_past = %d / %d:\n",
                speech.size(), t3->n_past(), t3->max_context());
    for (int32_t t : speech) std::printf("%d ", t);
    std::printf("\n");

    // Sanity checks
    int failures = 0;
    for (int32_t t : speech) {
        if (t < 0 || t >= params.vocab_limit) {
            std::printf("FAIL: token id %d outside [0, %d)\n",
                        t, params.vocab_limit);
            ++failures;
        }
    }
    if (speech.size() > static_cast<size_t>(params.max_tokens)) {
        std::printf("FAIL: generated %zu > max_tokens=%d\n",
                    speech.size(), params.max_tokens);
        ++failures;
    }
    if (t3->n_past() > t3->max_context()) {
        std::printf("FAIL: n_past=%d exceeded max_context=%d\n",
                    t3->n_past(), t3->max_context());
        ++failures;
    }
    if (failures > 0) {
        std::printf("\n%d smoke check(s) failed.\n", failures);
        return 1;
    }
    std::printf("\nSmoke test PASS.\n");
    return 0;
}
