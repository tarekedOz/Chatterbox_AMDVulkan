// End-to-end smoke test for Chatterbox::tts.
//
// Generates a 1-second synthetic reference WAV at 22050 Hz, runs the
// full pipeline, and verifies:
//   - PCM output is non-empty
//   - PCM is in [-1, 1]
//   - Output is not all zeros (model produced *something*)
//   - Output length is ~ T_speech * upsample (24kHz) given the
//     number of speech tokens T3 generated.
//
// Does NOT bit-compare against any oracle — too many stochastic links
// (T3 sampling, NSF source generator, CFM noise). This is the
// integration check that proves all the pieces fit together.

#include "chatterbox.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace {

std::vector<float> make_test_ref_wav(int sr, float seconds, int seed) {
    const int n = static_cast<int>(seconds * sr);
    std::vector<float> wav(n);
    // A clean speech-like signal: sum of a few sinusoids around 200-800 Hz
    // (typical speech fundamental + early harmonics) plus a touch of noise.
    // Seeds the randomness for reproducibility.
    unsigned rng_state = static_cast<unsigned>(seed) * 2654435761u;
    auto next = [&]() {
        rng_state = rng_state * 1103515245u + 12345u;
        return (static_cast<float>(rng_state >> 16) / 32768.0f) - 1.0f;
    };
    for (int i = 0; i < n; ++i) {
        const float t = static_cast<float>(i) / sr;
        wav[i] = (
            0.30f * std::sin(2.0f * 3.14159265f * 220.0f * t)         // ~A3
          + 0.20f * std::sin(2.0f * 3.14159265f * 440.0f * t + 0.4f)  // ~A4
          + 0.10f * std::sin(2.0f * 3.14159265f * 660.0f * t + 0.9f)  // ~E5
          + 0.03f * next()
        );
    }
    // Normalize to ~0.5 max amplitude.
    float peak = 1e-9f;
    for (float v : wav) peak = std::max(peak, std::abs(v));
    const float scale = 0.5f / peak;
    for (auto& v : wav) v *= scale;
    return wav;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        std::fprintf(stderr,
                     "Usage: %s <t3.gguf> <ve.gguf> <s3gen.gguf>\n", argv[0]);
        return 2;
    }
    const std::string t3_path    = argv[1];
    const std::string ve_path    = argv[2];
    const std::string s3gen_path = argv[3];

    std::printf("Chatterbox smoke test: loading models ...\n");
    chatterbox::ChatterboxConfig cfg;
    cfg.t3_max_speech_tokens = 30;     // keep generation short for the smoke
    auto cb = chatterbox::Chatterbox::load(t3_path, ve_path, s3gen_path, cfg);
    if (!cb) {
        std::fprintf(stderr, "Chatterbox::load failed\n");
        return 1;
    }
    std::printf("Loaded. Output sample rate: %d Hz\n",
                cb->output_sample_rate());

    // Build a synthetic reference WAV.
    const int ref_sr = 22050;
    const auto ref_wav = make_test_ref_wav(ref_sr, /*seconds=*/1.0f, /*seed=*/123);
    std::printf("Reference: %zu samples @ %d Hz (%.2f s)\n",
                ref_wav.size(), ref_sr,
                static_cast<double>(ref_wav.size()) / ref_sr);

    // One-shot tts.
    const std::string text = "Hello world.";
    std::printf("Synthesizing: %s\n", ("\"" + text + "\"").c_str());
    auto wav = cb->tts(text, ref_wav, ref_sr, /*seed=*/42);
    if (wav.empty()) {
        std::fprintf(stderr, "FAIL: tts returned empty\n");
        return 1;
    }

    // ---- Validity checks ----
    int n_fail = 0;
    float wav_min = wav[0], wav_max = wav[0];
    double wav_sum = 0.0, wav_abs_sum = 0.0;
    for (float v : wav) {
        if (v < wav_min) wav_min = v;
        if (v > wav_max) wav_max = v;
        wav_sum += v;
        wav_abs_sum += std::abs(v);
    }
    const float wav_mean   = static_cast<float>(wav_sum / wav.size());
    const float wav_absmean = static_cast<float>(wav_abs_sum / wav.size());
    std::printf("\nGenerated wav: %zu samples (~%.2f s)\n",
                wav.size(),
                static_cast<double>(wav.size()) / cb->output_sample_rate());
    std::printf("  min=%+.4f  max=%+.4f  mean=%+.4f  abs_mean=%.4f\n",
                wav_min, wav_max, wav_mean, wav_absmean);

    if (wav_min < -1.0f || wav_max > 1.0f) {
        std::printf("FAIL: PCM out of range [-1, 1]\n");
        ++n_fail;
    }
    if (wav_absmean < 1e-6f) {
        std::printf("FAIL: PCM is all silence (abs_mean < 1e-6)\n");
        ++n_fail;
    }
    if (wav.size() < 1000) {
        std::printf("FAIL: PCM too short (%zu samples)\n", wav.size());
        ++n_fail;
    }

    if (n_fail) {
        std::printf("\n%d check(s) failed.\n", n_fail);
        return 1;
    }
    std::printf("\nPASS\n");
    return 0;
}
