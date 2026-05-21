// S3 mel-extraction parity test.
//
// Reads tests/s3_mel_reference.bin (produced by scripts/reference_s3_mel.py),
// runs MelExtractor::log_mel against the embedded synthetic waveform,
// and compares element-wise to the NumPy oracle.
//
// Tolerance is tight (atol=1e-4 rtol=1e-3): the only sources of drift are
// (a) the cos/sin tables in our naive DFT vs numpy's mixed-radix FFT
// internals, and (b) f32 accumulation order in the filterbank matmul.
// Both should fit well under this bound for a 1 s signal.

#include "mel.h"
#include "model.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr,
                     "Usage: %s <s3gen.gguf> <s3_mel_reference.bin>\n", argv[0]);
        return 2;
    }

    auto m = chatterbox::load_model(argv[1]);
    if (!m) return 1;

    auto mel = chatterbox::MelExtractor::load(m.get());
    if (!mel) {
        std::fprintf(stderr, "Failed to construct MelExtractor.\n");
        return 1;
    }

    std::ifstream in(argv[2], std::ios::binary);
    if (!in) {
        std::fprintf(stderr, "Cannot open %s\n", argv[2]);
        return 1;
    }
    int32_t n_samples = 0;
    in.read(reinterpret_cast<char*>(&n_samples), sizeof(n_samples));
    std::vector<float> audio(n_samples);
    in.read(reinterpret_cast<char*>(audio.data()),
            n_samples * sizeof(float));
    int32_t exp_n_mels = 0, exp_n_frames = 0;
    in.read(reinterpret_cast<char*>(&exp_n_mels),   sizeof(exp_n_mels));
    in.read(reinterpret_cast<char*>(&exp_n_frames), sizeof(exp_n_frames));
    std::vector<float> expected(
        static_cast<size_t>(exp_n_mels) * exp_n_frames);
    in.read(reinterpret_cast<char*>(expected.data()),
            expected.size() * sizeof(float));
    if (!in) {
        std::fprintf(stderr, "Reference file too short.\n");
        return 1;
    }
    std::printf("Reference: %d samples, mel (%d, %d)\n",
                n_samples, exp_n_mels, exp_n_frames);

    int actual_n_mels = 0, actual_n_frames = 0;
    auto actual = mel->log_mel(audio, actual_n_mels, actual_n_frames);
    std::printf("Actual:    mel (%d, %d)\n",
                actual_n_mels, actual_n_frames);

    if (actual_n_mels != exp_n_mels || actual_n_frames != exp_n_frames) {
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

    const float ATOL = 1e-4f;
    const float RTOL = 1e-3f;
    float max_abs_err = 0.0f;
    size_t worst = 0;
    size_t failing = 0;
    for (size_t i = 0; i < expected.size(); ++i) {
        const float diff = std::abs(actual[i] - expected[i]);
        const float thr = ATOL + RTOL * std::abs(expected[i]);
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
