// S3Gen mel-extraction parity test.
//
// Two-stage check:
//   1. Filterbank check: load librosa's Slaney mel filterbank from
//      the side-dumped binary and compare element-wise to our
//      hand-rolled C++ generation. If this fails, the bug is in
//      generate_slaney_mel_filterbank(), not the extraction itself.
//   2. Log-mel check: run S3GenMelExtractor::log_mel against the
//      reference WAV, compare element-wise to the NumPy oracle.
//
// Tolerances: tight (atol=1e-5 rtol=1e-4) for filterbank generation
// since both sides are fp32; slightly looser (atol=1e-3 rtol=1e-3)
// for log-mel because the naive DFT in C++ accumulates differently
// from numpy's mixed-radix FFT.

#include "s3gen_mel.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr,
                     "Usage: %s <s3gen_mel_filters.bin> <s3gen_mel_reference.bin>\n",
                     argv[0]);
        return 2;
    }

    auto ext = chatterbox::S3GenMelExtractor::create();
    if (!ext) {
        std::fprintf(stderr, "Failed to construct S3GenMelExtractor.\n");
        return 1;
    }
    std::printf("S3GenMelExtractor: N_MELS=%d, N_FFT_BINS=%d, N_FFT=%d\n",
                chatterbox::S3GenMelExtractor::N_MELS,
                chatterbox::S3GenMelExtractor::N_FFT_BINS,
                chatterbox::S3GenMelExtractor::N_FFT);

    // ---- 1. Filterbank parity vs librosa ----
    std::ifstream fb_in(argv[1], std::ios::binary);
    if (!fb_in) {
        std::fprintf(stderr, "Cannot open filterbank ref: %s\n", argv[1]);
        return 1;
    }
    int32_t f_nm = 0, f_nb = 0;
    fb_in.read(reinterpret_cast<char*>(&f_nm), 4);
    fb_in.read(reinterpret_cast<char*>(&f_nb), 4);
    std::vector<float> libr_fb(static_cast<size_t>(f_nm) * f_nb);
    fb_in.read(reinterpret_cast<char*>(libr_fb.data()),
                libr_fb.size() * sizeof(float));
    std::printf("\nlibrosa filterbank: (%d, %d) loaded\n", f_nm, f_nb);

    const auto& cpp_fb = ext->mel_filterbank();
    if (cpp_fb.size() != libr_fb.size()) {
        std::printf("FAIL: filterbank size mismatch (%zu vs %zu)\n",
                    cpp_fb.size(), libr_fb.size());
        return 1;
    }

    float fb_max_err = 0.0f;
    size_t fb_worst = 0;
    size_t fb_failing = 0;
    const float FB_ATOL = 1e-5f, FB_RTOL = 1e-4f;
    for (size_t i = 0; i < libr_fb.size(); ++i) {
        const float diff = std::abs(cpp_fb[i] - libr_fb[i]);
        if (diff > fb_max_err) { fb_max_err = diff; fb_worst = i; }
        const float thr = FB_ATOL + FB_RTOL * std::abs(libr_fb[i]);
        if (diff > thr) ++fb_failing;
    }
    std::printf("Filterbank comparison (atol=%g, rtol=%g):\n",
                FB_ATOL, FB_RTOL);
    std::printf("  max abs err: %.4e (idx %zu: librosa=%+.5e cpp=%+.5e)\n",
                fb_max_err, fb_worst, libr_fb[fb_worst], cpp_fb[fb_worst]);
    std::printf("  failing:     %zu / %zu\n", fb_failing, libr_fb.size());
    if (fb_failing > 0) {
        std::printf("\nFAIL — hand-rolled Slaney filterbank doesn't match librosa\n");
        return 1;
    }

    // ---- 2. Log-mel parity ----
    std::ifstream in(argv[2], std::ios::binary);
    if (!in) {
        std::fprintf(stderr, "Cannot open mel ref: %s\n", argv[2]);
        return 1;
    }
    int32_t n_samples = 0;
    in.read(reinterpret_cast<char*>(&n_samples), 4);
    std::vector<float> audio(n_samples);
    in.read(reinterpret_cast<char*>(audio.data()),
            n_samples * sizeof(float));
    int32_t exp_nm = 0, exp_nf = 0;
    in.read(reinterpret_cast<char*>(&exp_nm), 4);
    in.read(reinterpret_cast<char*>(&exp_nf), 4);
    std::vector<float> expected(static_cast<size_t>(exp_nm) * exp_nf);
    in.read(reinterpret_cast<char*>(expected.data()),
            expected.size() * sizeof(float));
    std::printf("\nReference: %d samples (%.2f s), log-mel (%d, %d)\n",
                n_samples, static_cast<float>(n_samples) / 24000.0f,
                exp_nm, exp_nf);

    int act_nm = 0, act_nf = 0;
    auto actual = ext->log_mel(audio, act_nm, act_nf);
    std::printf("Actual:    log-mel (%d, %d)\n", act_nm, act_nf);

    if (act_nm != exp_nm || act_nf != exp_nf) {
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
    std::printf("Expected:  min=%+.4f max=%+.4f avg=%+.4f\n", emn, emx, eavg);
    std::printf("Actual:    min=%+.4f max=%+.4f avg=%+.4f\n", amn, amx, aavg);

    const float ATOL = 1e-3f, RTOL = 1e-3f;
    float max_err = 0.0f;
    size_t worst = 0;
    size_t failing = 0;
    for (size_t i = 0; i < expected.size(); ++i) {
        const float diff = std::abs(actual[i] - expected[i]);
        if (diff > max_err) { max_err = diff; worst = i; }
        const float thr = ATOL + RTOL * std::abs(expected[i]);
        if (diff > thr) ++failing;
    }
    std::printf("Comparison (atol=%g, rtol=%g):\n", ATOL, RTOL);
    std::printf("  max abs err: %.4e (idx %zu: expected=%+.4f actual=%+.4f)\n",
                max_err, worst, expected[worst], actual[worst]);
    std::printf("  failing:     %zu / %zu\n", failing, expected.size());
    if (failing > 0) {
        std::printf("\nFAILED\n");
        return 1;
    }
    std::printf("\nPASS\n");
    return 0;
}
