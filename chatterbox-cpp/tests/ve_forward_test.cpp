// VE forward parity test.
//
// Reads tests/ve_reference.bin (produced by scripts/reference_ve_forward.py),
// extracts the mel input + expected 256-d L2-normalized embedding, runs
// VE::forward on the C++ side, and compares element-wise.
//
// Tolerance: looser than T3's, since LSTM is iterative and fp16-vs-fp32
// errors compound over T*3 timesteps. atol=2e-2 rtol=2e-2. After
// L2-normalization values are in [-1, 1] so this is a fairly tight
// fraction of full range.

#include "model.h"
#include "ve.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "Usage: %s <ve.gguf> <ve_reference.bin>\n",
                     argv[0]);
        return 2;
    }

    auto m = chatterbox::load_model(argv[1]);
    if (!m) return 1;

    auto ve = chatterbox::VE::load(m.get());
    if (!ve) {
        std::fprintf(stderr, "Failed to construct VE.\n");
        return 1;
    }
    std::printf("VE loaded: %d-mel input, %d-d hidden, %d LSTM layers,\n"
                "           %d-d output, final_relu=%s\n",
                ve->config().n_mels, ve->config().hidden,
                ve->config().n_lstm_layers, ve->config().emb_dim,
                ve->config().final_relu ? "true" : "false");

    std::ifstream in(argv[2], std::ios::binary);
    if (!in) {
        std::fprintf(stderr, "Cannot open reference: %s\n", argv[2]);
        return 1;
    }
    int32_t n_frames = 0, n_mels = 0;
    in.read(reinterpret_cast<char*>(&n_frames), sizeof(n_frames));
    in.read(reinterpret_cast<char*>(&n_mels),   sizeof(n_mels));
    std::vector<float> mel(static_cast<size_t>(n_frames) * n_mels);
    in.read(reinterpret_cast<char*>(mel.data()), mel.size() * sizeof(float));
    int32_t emb_dim = 0;
    in.read(reinterpret_cast<char*>(&emb_dim), sizeof(emb_dim));
    std::vector<float> expected(emb_dim);
    in.read(reinterpret_cast<char*>(expected.data()),
            expected.size() * sizeof(float));
    if (!in) {
        std::fprintf(stderr, "Reference file too short.\n");
        return 1;
    }
    std::printf("\nReference: %d-frame mel (%d-d), %d-d embedding.\n",
                n_frames, n_mels, emb_dim);

    auto actual = ve->forward(mel, n_frames);
    if (actual.empty()) {
        std::fprintf(stderr, "VE::forward returned empty.\n");
        return 1;
    }
    if (actual.size() != expected.size()) {
        std::fprintf(stderr, "Size mismatch: actual=%zu expected=%zu\n",
                     actual.size(), expected.size());
        return 1;
    }

    double sumsq_actual = 0.0;
    for (float v : actual) sumsq_actual += static_cast<double>(v) * v;
    const float norm_actual = static_cast<float>(std::sqrt(sumsq_actual));
    std::printf("Actual emb norm: %.6f (should be ~1.0)\n", norm_actual);

    // Top-5 of each.
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
    for (int i : te) std::printf("(id=%d, v=%+.4f) ", i, expected[i]);
    std::printf("\nTop-5 actual:   ");
    for (int i : ta) std::printf("(id=%d, v=%+.4f) ", i, actual[i]);
    std::printf("\n\n");

    const float ATOL = 2e-2f;
    const float RTOL = 2e-2f;
    float max_abs_err = 0.0f;
    size_t worst_idx = 0;
    size_t failing = 0;
    for (size_t i = 0; i < expected.size(); ++i) {
        const float diff = std::abs(actual[i] - expected[i]);
        const float thr  = ATOL + RTOL * std::abs(expected[i]);
        if (diff > thr) ++failing;
        if (diff > max_abs_err) { max_abs_err = diff; worst_idx = i; }
    }
    std::printf("Comparison (atol=%g, rtol=%g):\n", ATOL, RTOL);
    std::printf("  max abs err: %.4e (idx %zu: expected=%+.4f actual=%+.4f)\n",
                max_abs_err, worst_idx,
                expected[worst_idx], actual[worst_idx]);
    std::printf("  failing:     %zu / %zu\n", failing, expected.size());

    if (te[0] != ta[0]) {
        std::printf("\nARGMAX MISMATCH: expected %d, got %d\n", te[0], ta[0]);
        return 1;
    }
    if (failing > 0) {
        std::printf("\nFAILED: %zu elements outside tolerance\n", failing);
        return 1;
    }
    std::printf("\nPASS\n");
    return 0;
}
