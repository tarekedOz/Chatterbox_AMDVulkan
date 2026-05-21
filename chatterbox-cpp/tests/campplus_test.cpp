// CAMPPlus speaker-encoder parity test.
//
// Reads tests/campplus_reference.bin (produced by
// scripts/reference_campplus.py), runs S3SpeakerEncoder::forward, and
// compares element-wise to the NumPy oracle.
//
// Tolerance: looser than T3 because CAMPPlus has more
// fp16-vs-fp32 drift sources (3 Dense blocks with 52 total layers,
// each with multiple BatchNorms + 1x1 convs). atol=5e-2 rtol=5e-2 is
// generous; tighten once we have data on the real distribution.

#include "model.h"
#include "s3spk.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr,
                     "Usage: %s <s3gen.gguf> <campplus_reference.bin>\n",
                     argv[0]);
        return 2;
    }

    auto m = chatterbox::load_model(argv[1]);
    if (!m) return 1;

    auto enc = chatterbox::S3SpeakerEncoder::load(m.get());
    if (!enc) {
        std::fprintf(stderr, "Failed to construct S3SpeakerEncoder.\n");
        return 1;
    }
    std::printf("S3SpeakerEncoder loaded.\n");

    std::ifstream in(argv[2], std::ios::binary);
    if (!in) {
        std::fprintf(stderr, "Cannot open %s\n", argv[2]);
        return 1;
    }
    int32_t T = 0, n_mels = 0;
    in.read(reinterpret_cast<char*>(&T),      sizeof(T));
    in.read(reinterpret_cast<char*>(&n_mels), sizeof(n_mels));
    std::vector<float> fbank(static_cast<size_t>(T) * n_mels);
    in.read(reinterpret_cast<char*>(fbank.data()),
            fbank.size() * sizeof(float));
    int32_t emb_dim = 0;
    in.read(reinterpret_cast<char*>(&emb_dim), sizeof(emb_dim));
    std::vector<float> expected(emb_dim);
    in.read(reinterpret_cast<char*>(expected.data()),
            expected.size() * sizeof(float));
    if (!in) {
        std::fprintf(stderr, "Reference file too short.\n");
        return 1;
    }
    std::printf("\nReference: fbank (%d, %d), embedding (%d,)\n",
                T, n_mels, emb_dim);

    auto actual = enc->forward(fbank, T, n_mels);
    if (actual.empty()) {
        std::fprintf(stderr, "Forward returned empty.\n");
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

    auto top5 = [&](const std::vector<float>& v) {
        std::vector<int> idx(v.size());
        for (int i = 0; i < (int)v.size(); ++i) idx[i] = i;
        std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(),
                          [&](int a, int b) { return std::abs(v[a]) > std::abs(v[b]); });
        idx.resize(5);
        return idx;
    };
    auto te = top5(expected);
    auto ta = top5(actual);
    std::printf("Top-5 |dims| expected: ");
    for (int i : te) std::printf("(%d, %+.4f) ", i, expected[i]);
    std::printf("\nTop-5 |dims| actual:   ");
    for (int i : ta) std::printf("(%d, %+.4f) ", i, actual[i]);
    std::printf("\n\n");

    const float ATOL = 5e-2f;
    const float RTOL = 5e-2f;
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

    if (failing > 0) {
        std::printf("\nFAILED\n");
        return 1;
    }
    std::printf("\nPASS\n");
    return 0;
}
