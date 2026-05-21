// VE 40-mel parity test against librosa via scripts/reference_ve_mel.py.

#include "ve_mel.h"

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
    mn = mx = v[0]; double s = 0.0;
    for (float x : v) { if (x<mn) mn=x; if (x>mx) mx=x; s+=x; }
    avg = static_cast<float>(s / v.size());
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "Usage: %s <ve_mel_reference.bin>\n", argv[0]);
        return 2;
    }
    std::ifstream f(argv[1], std::ios::binary);
    if (!f) return 1;

    int32_t n_samp, n_mels, n_frames;
    if (!read_exact(f, &n_samp, 4)) return 1;
    if (!read_exact(f, &n_mels, 4)) return 1;
    if (!read_exact(f, &n_frames, 4)) return 1;
    std::vector<float> audio(n_samp);
    if (!read_exact(f, audio.data(), n_samp * sizeof(float))) return 1;
    std::vector<float> expected(static_cast<size_t>(n_frames) * n_mels);
    if (!read_exact(f, expected.data(), expected.size() * sizeof(float))) return 1;
    std::printf("Ref: %d samp -> (%d, %d) mel\n", n_samp, n_frames, n_mels);

    auto ext = chatterbox::VEMelExtractor::create();
    int act_T, act_M;
    auto actual = ext->compute(audio, act_T, act_M);
    if (act_T != n_frames || act_M != n_mels) {
        std::printf("FAIL: shape (%d, %d) != (%d, %d)\n",
                    act_T, act_M, n_frames, n_mels);
        return 1;
    }

    float emn, emx, eavg, amn, amx, aavg;
    stats(expected, emn, emx, eavg);
    stats(actual,    amn, amx, aavg);
    std::printf("expected: min=%+.4f max=%+.4f avg=%+.4f\n", emn, emx, eavg);
    std::printf("actual:   min=%+.4f max=%+.4f avg=%+.4f\n", amn, amx, aavg);
    const float ATOL = 1e-3f, RTOL = 1e-3f;
    float max_err = 0.0f; size_t worst = 0; size_t failing = 0;
    for (size_t i = 0; i < expected.size(); ++i) {
        const float d = std::abs(actual[i] - expected[i]);
        if (d > max_err) { max_err = d; worst = i; }
        if (d > ATOL + RTOL * std::abs(expected[i])) ++failing;
    }
    std::printf("max abs err: %.4e  failing: %zu / %zu\n",
                max_err, failing, expected.size());
    if (failing > 0) { std::printf("FAIL\n"); return 1; }
    std::printf("PASS\n");
    return 0;
}
