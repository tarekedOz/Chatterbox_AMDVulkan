// Flow encoder parity test.
//
// Loads scripts/reference_flow_encoder.py's binary dump and compares the
// C++ FlowEncoder output (and every intermediate stage) against it.
//
// Stage tolerances are tight (atol=1e-3 rtol=1e-3 for floats up to ~5,
// looser for the inner-encoder stages where residual stacking amplifies
// fp16-vs-fp32 conversion drift). The encoder_proj output — what the CFM
// decoder consumes — is the strict check.

#include "model.h"
#include "flow_encoder.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace {

struct StageRef {
    std::string name;
    int T = 0, C = 0;
    std::vector<float> data;          // row-major (T, C)
};

struct Reference {
    int T_in = 0;
    int d_in = 0, d_out = 0;
    int n_stages = 0;
    std::vector<int32_t> tokens;
    std::vector<float>   spk_normed;        // (192,)
    std::vector<float>   spk_affine_out;    // (80,)
    std::vector<StageRef> stages;
};

bool read_exact(std::ifstream& f, void* buf, size_t n) {
    f.read(static_cast<char*>(buf), n);
    return f && static_cast<size_t>(f.gcount()) == n;
}

bool load_reference(const std::string& path, Reference& R) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); return false; }
    if (!read_exact(f, &R.T_in,     4)) return false;
    if (!read_exact(f, &R.d_in,     4)) return false;
    if (!read_exact(f, &R.d_out,    4)) return false;
    if (!read_exact(f, &R.n_stages, 4)) return false;

    R.tokens.resize(R.T_in);
    if (!read_exact(f, R.tokens.data(), R.T_in * sizeof(int32_t))) return false;
    R.spk_normed.resize(192);
    if (!read_exact(f, R.spk_normed.data(), 192 * sizeof(float))) return false;
    R.spk_affine_out.resize(80);
    if (!read_exact(f, R.spk_affine_out.data(), 80 * sizeof(float))) return false;

    R.stages.resize(R.n_stages);
    for (int i = 0; i < R.n_stages; ++i) {
        char nm[32] = {0};
        if (!read_exact(f, nm, 32)) return false;
        // strip trailing NULs
        size_t L = 0;
        while (L < 32 && nm[L] != '\0') ++L;
        R.stages[i].name.assign(nm, L);
        int32_t d0, d1, d2;
        if (!read_exact(f, &d0, 4)) return false;
        if (!read_exact(f, &d1, 4)) return false;
        if (!read_exact(f, &d2, 4)) return false;
        if (d0 != 1) {
            std::fprintf(stderr, "stage %s: expected dim0=1, got %d\n",
                          R.stages[i].name.c_str(), d0);
            return false;
        }
        R.stages[i].T = d1;
        R.stages[i].C = d2;
        const size_t n = static_cast<size_t>(d0) * d1 * d2;
        R.stages[i].data.resize(n);
        if (!read_exact(f, R.stages[i].data.data(), n * sizeof(float))) return false;
    }
    return true;
}

struct CompareResult {
    float    max_err = 0.0f;
    size_t   worst   = 0;
    size_t   failing = 0;
};

CompareResult compare(const std::vector<float>& expected,
                       const std::vector<float>& actual,
                       float atol, float rtol) {
    CompareResult r;
    const size_t n = std::min(expected.size(), actual.size());
    for (size_t i = 0; i < n; ++i) {
        const float diff = std::abs(actual[i] - expected[i]);
        if (diff > r.max_err) { r.max_err = diff; r.worst = i; }
        const float thr = atol + rtol * std::abs(expected[i]);
        if (diff > thr) ++r.failing;
    }
    return r;
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
    if (argc != 3) {
        std::fprintf(stderr,
                     "Usage: %s <s3gen.gguf> <flow_encoder_reference.bin>\n",
                     argv[0]);
        return 2;
    }
    Reference R;
    if (!load_reference(argv[2], R)) return 1;
    std::printf("Reference: T_in=%d, d_in=%d, d_out=%d, %d stages\n",
                R.T_in, R.d_in, R.d_out, R.n_stages);

    auto model = chatterbox::load_model(argv[1]);
    if (!model) { std::fprintf(stderr, "load_model failed\n"); return 1; }
    auto fe = chatterbox::FlowEncoder::load(model.get());
    if (!fe) { std::fprintf(stderr, "FlowEncoder::load failed\n"); return 1; }

    // ---- Speaker affine ----
    auto spk_aff = fe->affine_speaker(R.spk_normed);
    std::printf("\n[1/2] spk_embed_affine_layer parity:\n");
    {
        float emn, emx, eavg, amn, amx, aavg;
        stats(R.spk_affine_out, emn, emx, eavg);
        stats(spk_aff,           amn, amx, aavg);
        std::printf("  expected: min=%+.4f max=%+.4f avg=%+.4f\n", emn, emx, eavg);
        std::printf("  actual:   min=%+.4f max=%+.4f avg=%+.4f\n", amn, amx, aavg);
        auto r = compare(R.spk_affine_out, spk_aff, 1e-4f, 1e-4f);
        std::printf("  max abs err: %.4e  (idx %zu)  failing: %zu / %zu\n",
                    r.max_err, r.worst, r.failing, R.spk_affine_out.size());
        if (r.failing > 0) {
            std::printf("FAIL: spk_affine\n");
            return 1;
        }
    }

    // ---- Encoder forward + stage parity ----
    std::printf("\n[2/2] FlowEncoder.forward_with_stages parity:\n");
    std::unordered_map<std::string, std::vector<float>> stages;
    std::unordered_map<std::string, std::pair<int, int>> shapes;
    auto enc_out = fe->forward_with_stages(R.tokens, stages, shapes);
    if (enc_out.empty()) {
        std::fprintf(stderr, "forward_with_stages returned empty\n");
        return 1;
    }

    // Compare each stage (in the order they appear in the reference,
    // which is also the order they appear in the forward pass).
    //
    // Tolerances per-stage. The transformer residual stream amplifies
    // small fp16-quantization drifts of the weights linearly with depth;
    // by encoder block 5 we expect ~1e-1 absolute drift on values of
    // O(50). After after_norm the LN re-normalizes and the encoder_proj
    // output is back to O(1) where we tighten the tolerance.
    struct Tol { float atol, rtol; };
    // Observed maxima (after fp16 weight cast vs fp32 numpy oracle):
    //   after_input_embedding   4.8e-4   (fp16 quantization of embedding table)
    //   after_embed             1.7e-4
    //   after_prelookahead      1.7e-3
    //   after_enc_block0        1.7e-3
    //   after_enc_block5        5.0e-3
    //   after_uplayer           9.4e-3
    //   after_upembed           8.6e-3
    //   after_upenc_block0      9.7e-3
    //   after_upenc_block3      1.2e-2
    //   after_afternorm         6.9e-4
    //   after_encoderproj       6.3e-4
    // Tolerances are set ~3x the observed maxima to leave headroom for
    // minor compiler/SIMD variations between platforms.
    auto tol_for = [](const std::string& n) -> Tol {
        if (n == "after_input_embedding") return {2e-3f, 1e-3f};
        if (n == "after_embed")             return {2e-3f, 1e-3f};
        if (n == "after_prelookahead")      return {5e-3f, 2e-3f};
        if (n == "after_enc_block0")        return {1e-2f, 2e-3f};
        if (n == "after_enc_block5")        return {2e-2f, 2e-3f};
        if (n == "after_uplayer")           return {3e-2f, 2e-3f};
        if (n == "after_upembed")           return {3e-2f, 2e-3f};
        if (n == "after_upenc_block0")      return {3e-2f, 2e-3f};
        if (n == "after_upenc_block3")      return {4e-2f, 2e-3f};
        if (n == "after_afternorm")         return {3e-3f, 1e-3f};
        if (n == "after_encoderproj")       return {3e-3f, 1e-3f};
        return {1e-3f, 1e-3f};
    };

    int n_pass = 0, n_fail = 0;
    for (const auto& s : R.stages) {
        auto it = stages.find(s.name);
        if (it == stages.end()) {
            std::printf("  MISSING stage %s\n", s.name.c_str());
            ++n_fail;
            continue;
        }
        auto sh = shapes.at(s.name);
        const int act_T = sh.first;
        const int act_C = sh.second;
        Tol tol = tol_for(s.name);
        auto r = compare(s.data, it->second, tol.atol, tol.rtol);
        float emn, emx, eavg, amn, amx, aavg;
        stats(s.data,    emn, emx, eavg);
        stats(it->second, amn, amx, aavg);
        const char* mark = (r.failing == 0) ? "PASS" : "FAIL";
        std::printf("  %-22s exp(T,C)=(%d,%d)  act=(%d,%d)  "
                     "max|err|=%.3e  failing=%zu/%zu  "
                     "atol=%.0e rtol=%.0e  [%s]\n",
                     s.name.c_str(), s.T, s.C, act_T, act_C,
                     r.max_err, r.failing, s.data.size(),
                     tol.atol, tol.rtol, mark);
        std::printf("    expected: min=%+.4f max=%+.4f avg=%+.4f\n",
                     emn, emx, eavg);
        std::printf("    actual:   min=%+.4f max=%+.4f avg=%+.4f\n",
                     amn, amx, aavg);
        if (r.failing == 0) ++n_pass; else ++n_fail;
    }
    std::printf("\nstages: %d pass, %d fail\n", n_pass, n_fail);
    if (n_fail > 0) {
        std::printf("\nFAIL\n");
        return 1;
    }
    std::printf("\nPASS\n");
    return 0;
}
