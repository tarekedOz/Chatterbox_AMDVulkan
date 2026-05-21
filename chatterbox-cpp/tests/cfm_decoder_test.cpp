// CFM decoder parity test.
//
// Two checks:
//   1. estimator_forward at a fixed (t, r) probe: stage-by-stage compare
//      against scripts/reference_cfm_decoder.py.
//   2. solve_meanflow: full 2-step Euler with deterministic noise.

#include "model.h"
#include "cfm_decoder.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

struct StageRef {
    std::string name;
    int T = 0, C = 0;
    std::vector<float> data;
};

struct Reference {
    int T_mel = 0, d_mel = 0, d_in = 0, d_time = 0;
    int n_stages = 0, n_timesteps = 0;
    std::vector<float> mu;       // (T, 80) row-major (numpy)
    std::vector<float> spks;     // (80,)
    std::vector<float> cond;     // (T, 80)
    std::vector<float> mask;     // (T,)
    float t_probe = 0.0f, r_probe = 0.0f;
    std::vector<float> z;        // (T, 80) initial noise
    std::vector<StageRef> stages;
    std::vector<float> solver_final;   // (T, 80)
};

bool read_exact(std::ifstream& f, void* buf, size_t n) {
    f.read(static_cast<char*>(buf), n);
    return f && static_cast<size_t>(f.gcount()) == n;
}

bool load_reference(const std::string& path, Reference& R) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); return false; }

    if (!read_exact(f, &R.T_mel,       4)) return false;
    if (!read_exact(f, &R.d_mel,       4)) return false;
    if (!read_exact(f, &R.d_in,        4)) return false;
    if (!read_exact(f, &R.d_time,      4)) return false;
    if (!read_exact(f, &R.n_stages,    4)) return false;
    if (!read_exact(f, &R.n_timesteps, 4)) return false;

    const size_t TC = static_cast<size_t>(R.T_mel) * R.d_mel;
    R.mu.resize(TC);   if (!read_exact(f, R.mu.data(),   TC * sizeof(float))) return false;
    R.spks.resize(R.d_mel);
    if (!read_exact(f, R.spks.data(), R.d_mel * sizeof(float))) return false;
    R.cond.resize(TC); if (!read_exact(f, R.cond.data(), TC * sizeof(float))) return false;
    R.mask.resize(R.T_mel);
    if (!read_exact(f, R.mask.data(), R.T_mel * sizeof(float))) return false;
    if (!read_exact(f, &R.t_probe, 4)) return false;
    if (!read_exact(f, &R.r_probe, 4)) return false;
    R.z.resize(TC);    if (!read_exact(f, R.z.data(),    TC * sizeof(float))) return false;

    R.stages.resize(R.n_stages);
    for (int i = 0; i < R.n_stages; ++i) {
        char nm[32] = {0};
        if (!read_exact(f, nm, 32)) return false;
        size_t L = 0; while (L < 32 && nm[L] != '\0') ++L;
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

    // Solver section
    int solver_T = 0;
    if (!read_exact(f, &solver_T, 4)) return false;
    std::vector<float> dummy_z(TC);
    if (!read_exact(f, dummy_z.data(), TC * sizeof(float))) return false;  // same as R.z
    R.solver_final.resize(TC);
    if (!read_exact(f, R.solver_final.data(), TC * sizeof(float))) return false;
    (void) solver_T;
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
                     "Usage: %s <s3gen.gguf> <cfm_decoder_reference.bin>\n",
                     argv[0]);
        return 2;
    }
    Reference R;
    if (!load_reference(argv[2], R)) return 1;
    std::printf("Reference: T_mel=%d, d_mel=%d, d_in=%d, d_time=%d, "
                "%d stages, %d timesteps\n",
                R.T_mel, R.d_mel, R.d_in, R.d_time, R.n_stages, R.n_timesteps);
    std::printf("  t_probe=%.3f, r_probe=%.3f\n", R.t_probe, R.r_probe);

    auto model = chatterbox::load_model(argv[1]);
    if (!model) { std::fprintf(stderr, "load_model failed\n"); return 1; }
    auto dec = chatterbox::CFMDecoder::load(model.get());
    if (!dec) { std::fprintf(stderr, "CFMDecoder::load failed\n"); return 1; }

    // ---- [1/2] Estimator forward, stage-by-stage ----
    std::printf("\n[1/2] estimator_forward_with_stages (t=%.3f, r=%.3f, T=%d)\n",
                R.t_probe, R.r_probe, R.T_mel);
    std::unordered_map<std::string, std::vector<float>> stages;
    std::unordered_map<std::string, std::pair<int, int>> shapes;
    auto dxdt = dec->estimator_forward_with_stages(
        R.z, R.mask, R.mu, R.t_probe, R.spks, R.cond, R.r_probe, R.T_mel,
        stages, shapes);
    if (dxdt.empty()) {
        std::fprintf(stderr, "estimator_forward returned empty\n");
        return 1;
    }

    // Tolerances. The mid block tower amplifies fp16 quantization noise
    // linearly with depth; we use ~3x observed maxima as ceilings.
    struct Tol { float atol, rtol; };
    // Observed maxima (T_mel=16, fp16 weights vs fp32 numpy oracle):
    //   after_t_emb            6.8e-4
    //   after_r_emb            8.3e-4
    //   after_time_mixer       3.0e-3
    //   after_pack             0.0
    //   after_down0_resnet     5.0e-4
    //   after_down0_transformer 1.1e-3
    //   after_down0_downsample 1.7e-3
    //   after_mid0             2.5e-3
    //   after_mid11            1.1e-2
    //   after_up0_transformer  9.3e-2  (values up to 1138, so rel ~ 8e-5)
    //   after_up0_upsample     5.7e-2
    //   after_final_block      6.7e-3
    //   after_final_proj       2.0e-2
    //   solver_final           1.7e-2
    // Tolerances are ~3x the observed maxima for headroom across
    // platforms / SIMD variations.
    auto tol_for = [](const std::string& n) -> Tol {
        if (n == "after_t_emb")              return {3e-3f, 1e-3f};
        if (n == "after_r_emb")              return {3e-3f, 1e-3f};
        if (n == "after_time_mixer")         return {1e-2f, 2e-3f};
        if (n == "after_pack")                return {1e-4f, 1e-4f};
        if (n == "after_down0_resnet")       return {2e-3f, 1e-3f};
        if (n == "after_down0_transformer")  return {5e-3f, 1e-3f};
        if (n == "after_down0_downsample")   return {5e-3f, 1e-3f};
        if (n == "after_mid0")                return {1e-2f, 1e-3f};
        if (n == "after_mid11")               return {5e-2f, 2e-3f};
        if (n == "after_up0_transformer")    return {3e-1f, 2e-3f};
        if (n == "after_up0_upsample")        return {2e-1f, 2e-3f};
        if (n == "after_final_block")         return {2e-2f, 2e-3f};
        if (n == "after_final_proj")          return {6e-2f, 2e-3f};
        return {1e-2f, 5e-3f};
    };

    int n_pass = 0, n_fail = 0;
    for (const auto& s : R.stages) {
        auto it = stages.find(s.name);
        if (it == stages.end()) {
            std::printf("  MISSING stage %s\n", s.name.c_str());
            ++n_fail; continue;
        }
        auto sh = shapes.at(s.name);
        const int act_T = sh.first, act_C = sh.second;
        Tol tol = tol_for(s.name);
        auto r = compare(s.data, it->second, tol.atol, tol.rtol);
        float emn, emx, eavg, amn, amx, aavg;
        stats(s.data,    emn, emx, eavg);
        stats(it->second, amn, amx, aavg);
        const char* mark = (r.failing == 0) ? "PASS" : "FAIL";
        std::printf("  %-28s exp=(%d,%d)  act=(%d,%d)  "
                     "max|err|=%.3e  failing=%zu/%zu  [%s]\n",
                     s.name.c_str(), s.T, s.C, act_T, act_C,
                     r.max_err, r.failing, s.data.size(), mark);
        std::printf("    expected: min=%+.4f max=%+.4f avg=%+.4f\n",
                     emn, emx, eavg);
        std::printf("    actual:   min=%+.4f max=%+.4f avg=%+.4f\n",
                     amn, amx, aavg);
        if (r.failing == 0) ++n_pass; else ++n_fail;
    }
    std::printf("\nstages: %d pass, %d fail\n", n_pass, n_fail);

    // ---- [2/2] Meanflow solver ----
    std::printf("\n[2/2] solve_meanflow (%d timesteps)\n", R.n_timesteps);
    auto solver_out = dec->solve_meanflow(R.z, R.mu, R.mask, R.spks, R.cond,
                                            R.T_mel, R.n_timesteps);
    if (solver_out.empty()) {
        std::fprintf(stderr, "solve_meanflow returned empty\n");
        return 1;
    }
    {
        float emn, emx, eavg, amn, amx, aavg;
        stats(R.solver_final, emn, emx, eavg);
        stats(solver_out,     amn, amx, aavg);
        std::printf("  expected: min=%+.4f max=%+.4f avg=%+.4f\n", emn, emx, eavg);
        std::printf("  actual:   min=%+.4f max=%+.4f avg=%+.4f\n", amn, amx, aavg);
        auto r = compare(R.solver_final, solver_out, 5e-2f, 2e-3f);
        std::printf("  max abs err: %.4e  failing: %zu / %zu\n",
                    r.max_err, r.failing, R.solver_final.size());
        if (r.failing > 0) {
            std::printf("\nFAIL: solver output mismatch\n");
            return 1;
        }
    }

    if (n_fail > 0) {
        std::printf("\nFAIL\n");
        return 1;
    }
    std::printf("\nPASS\n");
    return 0;
}
