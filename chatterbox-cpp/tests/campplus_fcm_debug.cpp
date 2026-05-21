// Bisect helper: run ONLY the FCM head of CAMPPlus and compare against
// the numpy reference dumped by `scripts/reference_campplus.py --dump-fcm-bin`.
//
// The numpy ref produces (320, T) row-major fp32. The C++ side returns
// (T, 320) in ggml layout, which is the same memory order — element
// [c, t] in numpy at byte (c*T + t)*4; ggml's (T, 320) has element
// [t, c] at (t + c*T)*4. Identical layout.
//
// Inputs:
//   argv[1] = s3gen.gguf
//   argv[2] = campplus_fcm_reference.bin (the --dump-fcm-bin output)
//   argv[3] = campplus_reference.bin (for the fbank input)

#include "model.h"
#include "s3spk.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 4) {
        std::fprintf(stderr,
                     "Usage: %s <s3gen.gguf> <fcm_reference.bin> <full_reference.bin>\n",
                     argv[0]);
        return 2;
    }

    auto m = chatterbox::load_model(argv[1]);
    if (!m) return 1;
    auto enc = chatterbox::S3SpeakerEncoder::load(m.get());
    if (!enc) return 1;

    // Load fbank from full reference.
    std::ifstream fr(argv[3], std::ios::binary);
    int32_t T = 0, n_mels = 0;
    fr.read(reinterpret_cast<char*>(&T), 4);
    fr.read(reinterpret_cast<char*>(&n_mels), 4);
    std::vector<float> fbank(static_cast<size_t>(T) * n_mels);
    fr.read(reinterpret_cast<char*>(fbank.data()),
            fbank.size() * sizeof(float));

    // Load expected FCM from --dump-fcm-bin output.
    std::ifstream fcm_in(argv[2], std::ios::binary);
    int32_t exp_C = 0, exp_T = 0;
    fcm_in.read(reinterpret_cast<char*>(&exp_C), 4);
    fcm_in.read(reinterpret_cast<char*>(&exp_T), 4);
    std::vector<float> expected(static_cast<size_t>(exp_C) * exp_T);
    fcm_in.read(reinterpret_cast<char*>(expected.data()),
                expected.size() * sizeof(float));
    std::printf("Expected FCM: (%d, %d) = %zu elements\n",
                exp_C, exp_T, expected.size());

    int actual_C = 0, actual_T = 0;
    auto actual = enc->forward_fcm_only(fbank, T, n_mels, actual_C, actual_T);
    std::printf("Actual FCM:   (%d, %d) = %zu elements\n",
                actual_C, actual_T, actual.size());

    if (actual_C != exp_C || actual_T != exp_T) {
        std::printf("FAIL: shape mismatch\n");
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
    std::printf("Expected stats: min=%+.4f max=%+.4f avg=%+.4f\n", emn, emx, eavg);
    std::printf("Actual stats:   min=%+.4f max=%+.4f avg=%+.4f\n", amn, amx, aavg);

    // Element-wise diff
    float max_err = 0.0f;
    size_t worst_idx = 0;
    size_t failing = 0;
    const float ATOL = 1e-3f, RTOL = 1e-3f;
    for (size_t i = 0; i < expected.size(); ++i) {
        const float diff = std::abs(actual[i] - expected[i]);
        if (diff > max_err) { max_err = diff; worst_idx = i; }
        const float thr = ATOL + RTOL * std::abs(expected[i]);
        if (diff > thr) ++failing;
    }
    std::printf("Max abs err: %.4e (idx %zu: expected=%+.4f actual=%+.4f)\n",
                max_err, worst_idx, expected[worst_idx], actual[worst_idx]);
    std::printf("Failing (atol=%g rtol=%g): %zu / %zu\n",
                ATOL, RTOL, failing, expected.size());

    // Print a few elements side-by-side around channel 0, 100, 200, 300 at t=0
    std::printf("\nSample values (channel, t=0):\n");
    for (int c : {0, 50, 100, 200, 319}) {
        // Expected layout (C, T): byte offset (c * T + t) * 4 ; flat idx c*T + 0
        // Actual layout (T, 320): byte offset (t + c * T) * 4 ; same flat idx
        const size_t idx = c * exp_T;
        std::printf("  c=%3d: expected=%+.4f  actual=%+.4f  diff=%+.4e\n",
                    c, expected[idx], actual[idx], actual[idx] - expected[idx]);
    }

    if (failing == 0) {
        std::printf("\nPASS — FCM head matches numpy\n");
        return 0;
    }
    std::printf("\nFAIL — FCM head differs from numpy\n");
    return 1;
}
