// HiFTGenerator vocoder parity test.
//
// Three checks:
//   1. F0 predictor parity:    mel (T, 80) -> f0 (T,) via in-graph
//                              5 convs + ELU + Linear + abs.
//   2. Stage-by-stage decode:  conv_pre, ups0/1/2, src_rb0/1/2,
//                              src_add0/1/2, rb_avg0/1/2, refl_pad,
//                              conv_post, magnitude, phase.
//   3. Full waveform parity:   iSTFT(magnitude, phase) -> wav (T_wav,)
//
// The NSF source signal is passed in pre-computed (numpy reference
// generates it from the predicted F0 with seeded randomness). Both
// sides consume the same source, so the test is deterministic.

#include "model.h"
#include "hift_vocoder.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
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
    int T_mel = 0, d_mel = 0, n_fft = 0, hop_len = 0, upsample = 0;
    int T_wav = 0, T_stft = 0, n_stages = 0;
    std::vector<float> mel;        // (T, 80)
    std::vector<float> f0;         // (T,)
    std::vector<float> source;     // (T_wav,)
    std::vector<float> phase_vec;  // (9,)
    std::vector<float> noise_z;    // (9 * T_wav,)
    std::vector<float> s_stft;     // (T_stft, 18)
    std::vector<StageRef> stages;
    std::vector<float> wav;        // (T_wav,)
};

bool read_exact(std::ifstream& f, void* buf, size_t n) {
    f.read(static_cast<char*>(buf), n);
    return f && static_cast<size_t>(f.gcount()) == n;
}

bool load_reference(const std::string& path, Reference& R) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); return false; }

    if (!read_exact(f, &R.T_mel,    4)) return false;
    if (!read_exact(f, &R.d_mel,    4)) return false;
    if (!read_exact(f, &R.n_fft,    4)) return false;
    if (!read_exact(f, &R.hop_len,  4)) return false;
    if (!read_exact(f, &R.upsample, 4)) return false;
    if (!read_exact(f, &R.T_wav,    4)) return false;
    if (!read_exact(f, &R.T_stft,   4)) return false;
    if (!read_exact(f, &R.n_stages, 4)) return false;

    R.mel.resize(static_cast<size_t>(R.T_mel) * R.d_mel);
    if (!read_exact(f, R.mel.data(), R.mel.size() * sizeof(float))) return false;
    R.f0.resize(R.T_mel);
    if (!read_exact(f, R.f0.data(), R.f0.size() * sizeof(float))) return false;
    R.source.resize(R.T_wav);
    if (!read_exact(f, R.source.data(), R.source.size() * sizeof(float))) return false;
    R.phase_vec.resize(9);
    if (!read_exact(f, R.phase_vec.data(), 9 * sizeof(float))) return false;
    R.noise_z.resize(static_cast<size_t>(9) * R.T_wav);
    if (!read_exact(f, R.noise_z.data(), R.noise_z.size() * sizeof(float))) return false;
    const int two_bins = R.n_fft + 2;     // n_fft//2+1 * 2
    R.s_stft.resize(static_cast<size_t>(R.T_stft) * two_bins);
    if (!read_exact(f, R.s_stft.data(), R.s_stft.size() * sizeof(float))) return false;

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
        if (d0 != 1) return false;
        R.stages[i].T = d1;
        R.stages[i].C = d2;
        const size_t n = static_cast<size_t>(d1) * d2;
        R.stages[i].data.resize(n);
        if (!read_exact(f, R.stages[i].data.data(), n * sizeof(float))) return false;
    }
    R.wav.resize(R.T_wav);
    if (!read_exact(f, R.wav.data(), R.wav.size() * sizeof(float))) return false;
    return true;
}

struct CompareResult { float max_err = 0.0f; size_t worst = 0; size_t failing = 0; };

CompareResult compare(const std::vector<float>& exp,
                       const std::vector<float>& act,
                       float atol, float rtol) {
    CompareResult r;
    const size_t n = std::min(exp.size(), act.size());
    for (size_t i = 0; i < n; ++i) {
        const float diff = std::abs(act[i] - exp[i]);
        if (diff > r.max_err) { r.max_err = diff; r.worst = i; }
        const float thr = atol + rtol * std::abs(exp[i]);
        if (diff > thr) ++r.failing;
    }
    return r;
}

void stats(const std::vector<float>& v, float& mn, float& mx, float& avg) {
    if (v.empty()) { mn = mx = avg = 0.0f; return; }
    mn = mx = v[0];
    double s = 0.0;
    for (float x : v) {
        if (x < mn) mn = x; if (x > mx) mx = x; s += x;
    }
    avg = static_cast<float>(s / v.size());
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr,
                     "Usage: %s <s3gen.gguf> <hift_vocoder_reference.bin>\n",
                     argv[0]);
        return 2;
    }
    Reference R;
    if (!load_reference(argv[2], R)) return 1;
    std::printf("Reference: T_mel=%d, T_wav=%d, T_stft=%d, n_fft=%d, hop=%d, "
                "%d stages\n", R.T_mel, R.T_wav, R.T_stft, R.n_fft, R.hop_len,
                R.n_stages);

    auto model = chatterbox::load_model(argv[1]);
    if (!model) { std::fprintf(stderr, "load_model failed\n"); return 1; }
    auto v = chatterbox::HiFTVocoder::load(model.get());
    if (!v) { std::fprintf(stderr, "HiFTVocoder::load failed\n"); return 1; }

    // ---- [1/4] NSF source generator parity ----
    std::printf("\n[1/4] NSF source generator parity\n");
    {
        // Run with the SAME rng inputs as the numpy reference so the
        // output is bit-equivalent (within fp32 arithmetic).
        auto src = v->generate_source(R.f0, R.phase_vec, R.noise_z);
        if (src.empty()) { std::fprintf(stderr, "generate_source empty\n"); return 1; }
        float emn, emx, eavg, amn, amx, aavg;
        stats(R.source, emn, emx, eavg);
        stats(src,       amn, amx, aavg);
        std::printf("  expected: min=%+.4f max=%+.4f avg=%+.4f\n", emn, emx, eavg);
        std::printf("  actual:   min=%+.4f max=%+.4f avg=%+.4f\n", amn, amx, aavg);
        auto r = compare(R.source, src, 1e-5f, 1e-5f);
        std::printf("  max abs err: %.4e  failing: %zu / %zu\n",
                    r.max_err, r.failing, R.source.size());
        if (r.failing > 0) { std::printf("FAIL: NSF source\n"); return 1; }
    }

    // ---- [2/4] F0 predictor parity ----
    std::printf("\n[2/4] F0 predictor parity\n");
    {
        auto f0 = v->predict_f0(R.mel, R.T_mel);
        if (f0.empty()) { std::fprintf(stderr, "predict_f0 returned empty\n"); return 1; }
        float emn, emx, eavg, amn, amx, aavg;
        stats(R.f0, emn, emx, eavg);
        stats(f0,    amn, amx, aavg);
        std::printf("  expected: min=%+.4f max=%+.4f avg=%+.4f\n", emn, emx, eavg);
        std::printf("  actual:   min=%+.4f max=%+.4f avg=%+.4f\n", amn, amx, aavg);
        auto r = compare(R.f0, f0, 5e-4f, 5e-3f);
        std::printf("  max abs err: %.4e  failing: %zu / %zu\n",
                    r.max_err, r.failing, R.f0.size());
        if (r.failing > 0) { std::printf("FAIL: f0\n"); return 1; }
    }

    // ---- [3/4] Decode stage-by-stage parity ----
    std::printf("\n[3/4] decode_with_stages\n");
    std::unordered_map<std::string, std::vector<float>> stages;
    std::unordered_map<std::string, std::pair<int, int>> shapes;
    auto wav = v->decode_with_stages(R.mel, R.T_mel, R.source, R.T_wav,
                                        stages, shapes);
    if (wav.empty()) {
        std::fprintf(stderr, "decode returned empty\n");
        return 1;
    }

    // Per-stage tolerances. The conv_pre output sees fp16 weights with
    // ~5 layers of weight_norm fusion so we expect more drift than the
    // pure transformer modules.
    struct Tol { float atol, rtol; };
    auto tol_for = [](const std::string& n) -> Tol {
        if (n == "source_stft")           return {5e-5f, 5e-5f};
        if (n == "after_conv_pre")        return {3e-2f, 5e-3f};
        if (n.rfind("after_ups", 0) == 0) return {2e-2f, 5e-3f};
        if (n.rfind("after_src_rb", 0) == 0) return {1e-2f, 5e-3f};
        if (n.rfind("after_src_add", 0) == 0) return {3e-2f, 5e-3f};
        if (n.rfind("after_rb_avg", 0) == 0) return {1e-1f, 5e-3f};
        if (n == "after_refl_pad")        return {2e-2f, 5e-3f};
        if (n == "after_conv_post")       return {5e-2f, 5e-3f};
        if (n == "magnitude")             return {1e-1f, 5e-3f};
        if (n == "phase")                 return {5e-3f, 1e-3f};
        return {1e-2f, 5e-3f};
    };

    int n_pass = 0, n_fail = 0;
    for (const auto& s : R.stages) {
        auto it = stages.find(s.name);
        if (it == stages.end()) {
            std::printf("  MISSING %s\n", s.name.c_str());
            ++n_fail; continue;
        }
        auto sh = shapes.at(s.name);
        Tol tol = tol_for(s.name);
        auto r = compare(s.data, it->second, tol.atol, tol.rtol);
        float emn, emx, eavg, amn, amx, aavg;
        stats(s.data,    emn, emx, eavg);
        stats(it->second, amn, amx, aavg);
        const char* mark = (r.failing == 0) ? "PASS" : "FAIL";
        std::printf("  %-22s exp=(%d,%d)  act=(%d,%d)  max|err|=%.3e  "
                     "failing=%zu/%zu  [%s]\n",
                     s.name.c_str(), s.T, s.C, sh.first, sh.second,
                     r.max_err, r.failing, s.data.size(), mark);
        std::printf("    expected: min=%+.4f max=%+.4f avg=%+.4f\n",
                     emn, emx, eavg);
        std::printf("    actual:   min=%+.4f max=%+.4f avg=%+.4f\n",
                     amn, amx, aavg);
        if (r.failing == 0) ++n_pass; else ++n_fail;
    }
    std::printf("\nstages: %d pass, %d fail\n", n_pass, n_fail);

    // ---- [4/4] Final waveform parity ----
    std::printf("\n[4/4] Final waveform parity\n");
    {
        float emn, emx, eavg, amn, amx, aavg;
        stats(R.wav, emn, emx, eavg);
        stats(wav,    amn, amx, aavg);
        std::printf("  expected: min=%+.4f max=%+.4f avg=%+.4f\n", emn, emx, eavg);
        std::printf("  actual:   min=%+.4f max=%+.4f avg=%+.4f\n", amn, amx, aavg);
        auto r = compare(R.wav, wav, 5e-3f, 5e-3f);
        std::printf("  max abs err: %.4e  failing: %zu / %zu\n",
                    r.max_err, r.failing, R.wav.size());
        if (r.failing > 0) {
            std::printf("FAIL: waveform mismatch\n");
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
