// S3 audio tokenizer encoder parity test.
//
// Reads tests/s3_encoder_reference.bin (produced by
// scripts/reference_s3_encoder.py), runs S3Encoder::forward against the
// embedded log-mel, compares element-wise to the NumPy oracle.
//
// Tolerance: looser than T3 because (a) more layers (6 attention + 6 MLP),
// (b) RoPE adds trig accumulation, (c) FSMN's depthwise conv adds another
// fp16 weight path. atol=2e-2 rtol=2e-2 is fine for a 6-layer encoder
// running fp16 weights vs an fp32 oracle.

#include "model.h"
#include "s3enc.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr,
                     "Usage: %s <s3gen.gguf> <s3_encoder_reference.bin>\n",
                     argv[0]);
        return 2;
    }

    auto m = chatterbox::load_model(argv[1]);
    if (!m) return 1;

    auto enc = chatterbox::S3Encoder::load(m.get());
    if (!enc) {
        std::fprintf(stderr, "Failed to construct S3Encoder.\n");
        return 1;
    }
    std::printf("S3Encoder loaded: n_mels=%d, n_state=%d, n_head=%d, head_dim=%d,\n"
                "                  n_layer=%d, fsmn_k=%d\n",
                enc->config().n_mels, enc->config().n_state,
                enc->config().n_head, enc->config().head_dim,
                enc->config().n_layer, enc->config().fsmn_k);

    std::ifstream in(argv[2], std::ios::binary);
    if (!in) {
        std::fprintf(stderr, "Cannot open %s\n", argv[2]);
        return 1;
    }
    int32_t T_mel = 0, n_mels = 0;
    in.read(reinterpret_cast<char*>(&T_mel),  sizeof(T_mel));
    in.read(reinterpret_cast<char*>(&n_mels), sizeof(n_mels));
    std::vector<float> mel(static_cast<size_t>(n_mels) * T_mel);
    in.read(reinterpret_cast<char*>(mel.data()), mel.size() * sizeof(float));
    int32_t T_tok = 0, n_state = 0;
    in.read(reinterpret_cast<char*>(&T_tok),   sizeof(T_tok));
    in.read(reinterpret_cast<char*>(&n_state), sizeof(n_state));
    std::vector<float> expected(static_cast<size_t>(T_tok) * n_state);
    in.read(reinterpret_cast<char*>(expected.data()),
            expected.size() * sizeof(float));
    if (!in) {
        std::fprintf(stderr, "Reference file too short.\n");
        return 1;
    }
    std::printf("\nReference: log-mel (%d, %d), hidden (%d, %d)\n",
                n_mels, T_mel, T_tok, n_state);

    int actual_T_tok = 0, actual_n_state = 0;
    auto actual = enc->forward(mel, n_mels, T_mel, actual_T_tok, actual_n_state);
    if (actual.empty()) {
        std::fprintf(stderr, "S3Encoder::forward returned empty.\n");
        return 1;
    }
    std::printf("Actual:    hidden (%d, %d)\n", actual_T_tok, actual_n_state);
    if (actual_T_tok != T_tok || actual_n_state != n_state) {
        std::printf("FAIL: shape mismatch\n");
        return 1;
    }
    if (actual.size() != expected.size()) {
        std::printf("FAIL: size mismatch (%zu vs %zu)\n",
                    actual.size(), expected.size());
        return 1;
    }

    auto stats = [](const std::vector<float>& v) {
        float mn = v[0], mx = v[0];
        double s = 0.0;
        for (float x : v) { mn = std::min(mn, x); mx = std::max(mx, x); s += x; }
        return std::tuple<float, float, float>{mn, mx,
                                                static_cast<float>(s / v.size())};
    };
    auto [emn, emx, eavg] = stats(expected);
    auto [amn, amx, aavg] = stats(actual);
    std::printf("Expected: min=%+.4f max=%+.4f avg=%+.4f\n", emn, emx, eavg);
    std::printf("Actual:   min=%+.4f max=%+.4f avg=%+.4f\n", amn, amx, aavg);

    // Spot-check the first row's top-5 dims.
    auto top5_of = [&](const std::vector<float>& v, int t) {
        std::vector<std::pair<float, int>> pairs(n_state);
        for (int i = 0; i < n_state; ++i)
            pairs[i] = {v[static_cast<size_t>(t) * n_state + i], i};
        std::partial_sort(pairs.begin(), pairs.begin() + 5, pairs.end(),
                          std::greater<>());
        std::vector<int> out(5);
        for (int i = 0; i < 5; ++i) out[i] = pairs[i].second;
        return out;
    };
    auto te0 = top5_of(expected, 0);
    auto ta0 = top5_of(actual, 0);
    std::printf("Top-5 (row 0) expected: ");
    for (int i : te0) std::printf("%d ", i);
    std::printf("\nTop-5 (row 0) actual:   ");
    for (int i : ta0) std::printf("%d ", i);
    std::printf("\n\n");

    const float ATOL = 2e-2f;
    const float RTOL = 2e-2f;
    float max_abs_err = 0.0f;
    size_t worst = 0;
    size_t failing = 0;
    for (size_t i = 0; i < expected.size(); ++i) {
        const float diff = std::abs(actual[i] - expected[i]);
        const float thr  = ATOL + RTOL * std::abs(expected[i]);
        if (diff > thr) ++failing;
        if (diff > max_abs_err) { max_abs_err = diff; worst = i; }
    }
    std::printf("Comparison (atol=%g, rtol=%g):\n", ATOL, RTOL);
    std::printf("  max abs err: %.4e (idx %zu: expected=%+.4f actual=%+.4f)\n",
                max_abs_err, worst, expected[worst], actual[worst]);
    std::printf("  failing:     %zu / %zu\n", failing, expected.size());

    if (te0[0] != ta0[0]) {
        std::printf("\nARGMAX MISMATCH (row 0): expected %d, actual %d\n",
                    te0[0], ta0[0]);
        return 1;
    }
    if (failing > 0) {
        std::printf("\nFAILED: %zu elements outside tolerance\n", failing);
        return 1;
    }
    std::printf("\nPASS\n");
    return 0;
}
