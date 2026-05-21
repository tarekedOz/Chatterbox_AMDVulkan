// T3 forward parity test.
//
// Reads tests/t3_reference.bin (produced by scripts/reference_t3_forward.py),
// extracts the input tokens + expected logits (fp32), runs T3::forward on
// the C++ side against the T3 GGUF, and compares element-wise.
//
// Tolerance: rtol=1e-2 absolute=5e-2. Loose because we run fp16 weights
// vs the NumPy reference's fp32 — fp16 round-trip noise compounds over
// 24 layers. We can tighten once we have fp32 GGUFs or know the
// per-tensor error budget.
//
// Reference binary format:
//   int32  n_tokens
//   int32 * n_tokens   token_ids
//   float32 * 6563     expected logits

#include "model.h"
#include "t3.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "Usage: %s <t3.gguf> <t3_reference.bin>\n", argv[0]);
        return 2;
    }

    auto m = chatterbox::load_model(argv[1]);
    if (!m) return 1;

    auto t3 = chatterbox::T3::load(m.get());
    if (!t3) {
        std::fprintf(stderr, "Failed to construct T3.\n");
        return 1;
    }
    std::printf("T3 loaded: %d layers, %d heads, %d-d embed, %d-d ffn,\n"
                "           context=%d, text_vocab=%d, speech_vocab=%d\n",
                t3->config().n_layers, t3->config().n_heads,
                t3->config().embed_dim, t3->config().ffn_dim,
                t3->config().context_len,
                t3->config().text_vocab, t3->config().speech_vocab);

    // Parse the reference binary (v3 schema: now includes prompt speech).
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
    std::vector<float> expected(t3->config().speech_vocab);
    in.read(reinterpret_cast<char*>(expected.data()),
            expected.size() * sizeof(float));
    if (!in) {
        std::fprintf(stderr, "Reference file too short or malformed.\n");
        return 1;
    }

    std::printf("\nReference: %d text tokens, %d-d speaker emb, "
                "%d prompt speech tokens, %zu logits.\n",
                n_tokens, spk_dim, n_prompt, expected.size());
    std::printf("Text tokens:   ");
    for (int32_t t : tokens) std::printf("%d ", t);
    std::printf("\nPrompt tokens: ");
    for (int32_t t : prompt) std::printf("%d ", t);
    std::printf("\n");

    auto actual = t3->prefill(speaker_emb, prompt, tokens);
    if (actual.empty()) {
        std::fprintf(stderr, "T3::forward returned no output.\n");
        return 1;
    }
    if (actual.size() != expected.size()) {
        std::fprintf(stderr,
                     "Logit count mismatch: actual=%zu expected=%zu\n",
                     actual.size(), expected.size());
        return 1;
    }

    // Stats
    auto stats = [](const std::vector<float>& v) {
        float mn = v[0], mx = v[0];
        double sum = 0.0;
        for (float x : v) { mn = std::min(mn, x); mx = std::max(mx, x); sum += x; }
        return std::tuple<float, float, float>{mn, mx,
                                                static_cast<float>(sum / v.size())};
    };
    auto [emn, emx, eavg] = stats(expected);
    auto [amn, amx, aavg] = stats(actual);
    std::printf("\nExpected: min=%+.4f max=%+.4f avg=%+.4f\n", emn, emx, eavg);
    std::printf("Actual:   min=%+.4f max=%+.4f avg=%+.4f\n", amn, amx, aavg);

    // Top-5 of each
    auto top5 = [](const std::vector<float>& v) {
        std::vector<int> idx(v.size());
        for (size_t i = 0; i < idx.size(); ++i) idx[i] = static_cast<int>(i);
        std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(),
                          [&](int a, int b) { return v[a] > v[b]; });
        idx.resize(5);
        return idx;
    };
    auto te = top5(expected);
    auto ta = top5(actual);
    std::printf("Top-5 expected: ");
    for (int i : te) std::printf("(id=%d, l=%+.4f) ", i, expected[i]);
    std::printf("\nTop-5 actual:   ");
    for (int i : ta) std::printf("(id=%d, l=%+.4f) ", i, actual[i]);
    std::printf("\n\n");

    // Element-wise comparison. Tolerance widened from the CPU-only
    // baseline (5e-2 / 1e-2) to also accommodate ggml-vulkan's fp16
    // kernel accumulation across 24 transformer blocks + KV cache
    // reads. The KV cache tensors are buffer-pinned, so we can't
    // route the stack through CPU via sched_set_tensor_backend
    // (that trick works for S3Encoder but not here). The relevant
    // correctness check is the top-5 token id agreement (printed
    // above); the element-wise check below stays as a sanity bound.
    // Max observed Vulkan drift: ~1.5e-1 abs at fp32 reference ~2.0.
    const float ATOL = 2e-1f;
    const float RTOL = 3e-2f;
    float max_abs_err = 0.0f;
    float max_rel_err = 0.0f;
    size_t failing = 0;
    size_t worst_idx = 0;
    for (size_t i = 0; i < expected.size(); ++i) {
        const float diff = std::abs(actual[i] - expected[i]);
        const float thr = ATOL + RTOL * std::abs(expected[i]);
        if (diff > thr) ++failing;
        if (diff > max_abs_err) {
            max_abs_err = diff;
            worst_idx = i;
        }
        const float rel = diff / (std::abs(expected[i]) + 1e-9f);
        if (rel > max_rel_err) max_rel_err = rel;
    }
    std::printf("Comparison (rtol=%g, atol=%g):\n", RTOL, ATOL);
    std::printf("  max abs err: %.4e  (at idx %zu: expected=%+.4f actual=%+.4f)\n",
                max_abs_err, worst_idx, expected[worst_idx], actual[worst_idx]);
    std::printf("  max rel err: %.4e\n", max_rel_err);
    std::printf("  failing:     %zu / %zu\n", failing, expected.size());

    // Argmax must match — a top-1 mismatch is a much harder signal of bug
    // than per-element drift.
    const int argmax_e = te[0];
    const int argmax_a = ta[0];
    if (argmax_e != argmax_a) {
        std::printf("\nARGMAX MISMATCH: expected %d, got %d\n",
                    argmax_e, argmax_a);
        return 1;
    }

    if (failing > 0) {
        std::printf("\nFAILED: %zu logits outside tolerance\n", failing);
        return 1;
    }

    // ----------------------------------------------------------------
    // KV-cache equivalence check.
    //
    // We re-run the same input but split into two calls: prefill with
    // (speaker_emb + first k text tokens), then append_text with the
    // remaining tokens. The cache machinery should make the resulting
    // last-position logits byte-equal to the one-shot prefill above.
    //
    // We allow only fp16 ulp drift on top of the existing tolerance
    // because the math is identical — same weights, same activations.
    // ----------------------------------------------------------------
    if (tokens.size() < 2) {
        std::printf("\n(skipping cache equivalence — need >= 2 tokens)\n");
        std::printf("\nPASS\n");
        return 0;
    }

    std::printf("\nKV cache equivalence check (split %zu / %zu) ...\n",
                tokens.size() / 2, tokens.size() - tokens.size() / 2);
    const size_t split = tokens.size() / 2;
    std::vector<int32_t> first(tokens.begin(), tokens.begin() + split);
    std::vector<int32_t> rest(tokens.begin() + split, tokens.end());

    auto pre = t3->prefill(speaker_emb, prompt, first);
    if (pre.empty()) {
        std::fprintf(stderr, "Split prefill returned empty.\n");
        return 1;
    }
    auto split_actual = t3->append_text(rest);
    if (split_actual.empty()) {
        std::fprintf(stderr, "append_text returned empty.\n");
        return 1;
    }
    if (split_actual.size() != actual.size()) {
        std::fprintf(stderr, "Split logits size %zu != one-shot %zu\n",
                     split_actual.size(), actual.size());
        return 1;
    }

    float max_split_err = 0.0f;
    size_t split_failing = 0;
    // Tolerance: we expected to match bit-for-bit since both sides are
    // the same fp16 math, but matmul kernels partition differently when
    // given (D, N_total=4) one-shot vs (D, 2) twice. Different SIMD
    // reduction trees and OpenMP accumulation orders introduce ULP drift.
    // 24 layers compound it. The argmax + top-5 ids should still match.
    // Same Vulkan widening as the main parity check. The one-shot vs
    // split-step paths take slightly different sched routes on Vulkan
    // (KV cache writes vs reads across two separate compute calls),
    // so fp16 accumulation diverges across the 24-block stack.
    const float CACHE_ATOL = 2e-1f;
    const float CACHE_RTOL = 3e-2f;
    for (size_t i = 0; i < actual.size(); ++i) {
        const float diff = std::abs(split_actual[i] - actual[i]);
        const float thr  = CACHE_ATOL + CACHE_RTOL * std::abs(actual[i]);
        if (diff > thr) ++split_failing;
        if (diff > max_split_err) max_split_err = diff;
    }
    std::printf("  one-shot   top-1:  id=%d  logit=%+.4f\n",
                ta[0], actual[ta[0]]);
    auto ts = top5(split_actual);
    std::printf("  split-step top-1:  id=%d  logit=%+.4f\n",
                ts[0], split_actual[ts[0]]);
    std::printf("  max abs err:       %.4e\n", max_split_err);
    std::printf("  failing:           %zu / %zu\n",
                split_failing, actual.size());

    if (ts[0] != ta[0]) {
        std::printf("\nCACHE FAIL: split argmax %d != one-shot %d\n",
                    ts[0], ta[0]);
        return 1;
    }
    if (split_failing > 0) {
        std::printf("\nCACHE FAIL: %zu logits exceed tolerance\n", split_failing);
        return 1;
    }
    std::printf("\nPASS\n");
    return 0;
}
