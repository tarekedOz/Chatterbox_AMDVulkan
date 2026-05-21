// Resampler parity test against scripts/reference_resampler.py
// (scipy.signal.resample_poly with Kaiser(beta=8.6)).

#include "resample.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
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
        std::fprintf(stderr, "Usage: %s <resampler_reference.bin>\n", argv[0]);
        return 2;
    }
    std::ifstream f(argv[1], std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "cannot open %s\n", argv[1]);
        return 1;
    }

    int32_t in_sr = 0, in_len = 0;
    if (!read_exact(f, &in_sr, 4)) return 1;
    if (!read_exact(f, &in_len, 4)) return 1;
    std::vector<float> input(in_len);
    if (!read_exact(f, input.data(), in_len * sizeof(float))) return 1;

    int32_t n_targets = 0;
    if (!read_exact(f, &n_targets, 4)) return 1;
    std::printf("Reference: %d samples @ %d Hz, %d targets\n",
                in_len, in_sr, n_targets);

    int n_fail = 0;
    for (int i = 0; i < n_targets; ++i) {
        int32_t out_sr = 0, out_len = 0;
        if (!read_exact(f, &out_sr, 4)) return 1;
        if (!read_exact(f, &out_len, 4)) return 1;
        std::vector<float> expected(out_len);
        if (!read_exact(f, expected.data(), out_len * sizeof(float))) return 1;

        auto actual = chatterbox::resample_audio(input, in_sr, out_sr);
        float emn, emx, eavg, amn, amx, aavg;
        stats(expected, emn, emx, eavg);
        stats(actual,    amn, amx, aavg);
        std::printf("\nTarget %d -> %d Hz:\n", in_sr, out_sr);
        std::printf("  expected: %d samples, min=%+.4f max=%+.4f avg=%+.4f\n",
                    out_len, emn, emx, eavg);
        std::printf("  actual:   %zu samples, min=%+.4f max=%+.4f avg=%+.4f\n",
                    actual.size(), amn, amx, aavg);

        if (actual.size() != static_cast<size_t>(out_len)) {
            std::printf("  FAIL: length mismatch\n");
            ++n_fail;
            continue;
        }
        // For sinc-based resampling, tolerance is dominated by phase
        // alignment of the impulse response — we expect ~1e-4 per
        // sample.
        const float ATOL = 1e-3f;
        float max_err = 0.0f;
        size_t worst = 0, failing = 0;
        for (size_t k = 0; k < actual.size(); ++k) {
            const float d = std::abs(actual[k] - expected[k]);
            if (d > max_err) { max_err = d; worst = k; }
            if (d > ATOL) ++failing;
        }
        std::printf("  max abs err: %.4e (idx %zu)  failing: %zu / %zu\n",
                    max_err, worst, failing, actual.size());
        if (failing > 0) {
            ++n_fail;
            // Print a few elements around the worst index for context.
            const size_t lo = (worst > 4) ? worst - 4 : 0;
            const size_t hi = std::min<size_t>(actual.size(), worst + 5);
            std::printf("  Worst region:\n");
            for (size_t k = lo; k < hi; ++k) {
                std::printf("    [%zu] expected=%+.6f actual=%+.6f diff=%+.6f\n",
                             k, expected[k], actual[k],
                             actual[k] - expected[k]);
            }
        }
    }
    if (n_fail) {
        std::printf("\n%d target(s) failed.\n", n_fail);
        return 1;
    }
    std::printf("\nPASS\n");
    return 0;
}
