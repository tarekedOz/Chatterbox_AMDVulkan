// Kaldi-style 80-d fbank parity test against torchaudio.compliance.kaldi.fbank.

#include "kaldi_fbank.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <vector>

namespace {

bool read_exact(std::ifstream& f, void* buf, size_t n) {
    f.read(static_cast<char*>(buf), n);
    return f && static_cast<size_t>(f.gcount()) == n;
}

void stats(const std::vector<float>& v, float& mn, float& mx, float& avg) {
    if (v.empty()) { mn = mx = avg = 0.0f; return; }
    mn = mx = v[0];
    double s = 0.0;
    for (float x : v) {
        if (x < mn) mn = x;
        if (x > mx) mx = x;
        s += x;
    }
    avg = static_cast<float>(s / v.size());
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "Usage: %s <kaldi_fbank_reference.bin>\n", argv[0]);
        return 2;
    }
    std::ifstream f(argv[1], std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", argv[1]); return 1; }

    int32_t sr, n_samp, n_mels, n_frames;
    if (!read_exact(f, &sr,       4)) return 1;
    if (!read_exact(f, &n_samp,   4)) return 1;
    if (!read_exact(f, &n_mels,   4)) return 1;
    if (!read_exact(f, &n_frames, 4)) return 1;
    std::vector<float> audio(n_samp);
    if (!read_exact(f, audio.data(), n_samp * sizeof(float))) return 1;
    std::vector<float> expected(static_cast<size_t>(n_frames) * n_mels);
    if (!read_exact(f, expected.data(), expected.size() * sizeof(float))) return 1;
    std::printf("Reference: %d samples @ %d Hz -> (%d, %d) fbank\n",
                n_samp, sr, n_frames, n_mels);

    auto ext = chatterbox::KaldiFbankExtractor::create();
    int act_frames = 0, act_mels = 0;
    auto actual = ext->extract(audio, act_frames, act_mels);
    if (act_frames != n_frames || act_mels != n_mels) {
        std::printf("FAIL: shape mismatch (%d, %d) vs (%d, %d)\n",
                    act_frames, act_mels, n_frames, n_mels);
        return 1;
    }

    float emn, emx, eavg, amn, amx, aavg;
    stats(expected, emn, emx, eavg);
    stats(actual,    amn, amx, aavg);
    std::printf("expected: min=%+.4f max=%+.4f avg=%+.4f\n", emn, emx, eavg);
    std::printf("actual:   min=%+.4f max=%+.4f avg=%+.4f\n", amn, amx, aavg);

    // Tolerance: tight for the inner bins (torchaudio uses double internally
    // for some ops; fp32 drift typically <= 1e-3 per bin).
    const float ATOL = 1e-2f, RTOL = 1e-3f;
    float max_err = 0.0f; size_t worst = 0; size_t failing = 0;
    for (size_t i = 0; i < expected.size(); ++i) {
        const float d = std::abs(actual[i] - expected[i]);
        if (d > max_err) { max_err = d; worst = i; }
        const float thr = ATOL + RTOL * std::abs(expected[i]);
        if (d > thr) ++failing;
    }
    std::printf("max abs err: %.4e (idx %zu: expected=%+.4f actual=%+.4f)\n",
                max_err, worst,
                expected[worst], actual[worst]);
    std::printf("failing: %zu / %zu  (atol=%.0e rtol=%.0e)\n",
                failing, expected.size(), ATOL, RTOL);
    if (failing > 0) {
        // Print a row near the worst index for diagnosis.
        const int t = static_cast<int>(worst) / n_mels;
        const int b = static_cast<int>(worst) % n_mels;
        std::printf("Worst row t=%d, b=%d:\n", t, b);
        const float* exp_row = expected.data() + static_cast<size_t>(t) * n_mels;
        const float* act_row = actual.data()    + static_cast<size_t>(t) * n_mels;
        for (int j = std::max(0, b - 4); j < std::min<int>(n_mels, b + 5); ++j) {
            std::printf("  bin %2d: expected=%+.6f actual=%+.6f diff=%+.6f\n",
                         j, exp_row[j], act_row[j], act_row[j] - exp_row[j]);
        }
        return 1;
    }
    std::printf("\nPASS\n");
    return 0;
}
