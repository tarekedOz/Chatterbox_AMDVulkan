// End-to-end test with a REAL voice sample (Abigail.wav from
// devnen/Chatterbox-TTS-Server's voices/ directory).
//
// Loads the reference WAV at 44.1 kHz, runs full Chatterbox::tts with
// "Hello world." text, and writes the resulting 24 kHz PCM to disk so
// we can listen to it. This is the first end-to-end check using a
// real recording rather than synthetic test signals.

#include "chatterbox.h"
#include "wav_io.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace {

void stats(const std::vector<float>& v, float& mn, float& mx,
             float& avg, float& abs_mean) {
    if (v.empty()) { mn = mx = avg = abs_mean = 0.0f; return; }
    mn = mx = v[0]; double s = 0.0, as = 0.0;
    for (float x : v) {
        if (x < mn) mn = x;
        if (x > mx) mx = x;
        s += x; as += std::abs(x);
    }
    avg      = static_cast<float>(s / v.size());
    abs_mean = static_cast<float>(as / v.size());
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 6) {
        std::fprintf(stderr,
                     "Usage: %s <t3.gguf> <ve.gguf> <s3gen.gguf>\n"
                     "          <ref_voice.wav> <out_synth.wav>\n", argv[0]);
        return 2;
    }
    const std::string t3_path    = argv[1];
    const std::string ve_path    = argv[2];
    const std::string s3gen_path = argv[3];
    const std::string ref_wav    = argv[4];
    const std::string out_wav    = argv[5];

    std::printf("Loading reference WAV: %s\n", ref_wav.c_str());
    std::vector<float> ref_audio;
    int ref_sr = 0;
    if (!chatterbox::read_wav_mono(ref_wav, ref_audio, ref_sr)) {
        return 1;
    }
    std::printf("  %zu samples @ %d Hz (%.2f s)\n",
                ref_audio.size(), ref_sr,
                static_cast<double>(ref_audio.size()) / ref_sr);
    float rmn, rmx, ravg, rabs;
    stats(ref_audio, rmn, rmx, ravg, rabs);
    std::printf("  reference stats: min=%+.4f max=%+.4f abs_mean=%.4f\n",
                rmn, rmx, rabs);

    std::printf("\nLoading Chatterbox models ...\n");
    chatterbox::ChatterboxConfig cfg;
    cfg.t3_max_speech_tokens = 80;       // ~3.2 sec of generated audio
    auto cb = chatterbox::Chatterbox::load(t3_path, ve_path, s3gen_path, cfg);
    if (!cb) {
        std::fprintf(stderr, "Chatterbox::load failed\n");
        return 1;
    }

    const std::string text = "Hello, this is a test of the chatterbox text to speech system.";
    std::printf("\nSynthesizing: %s\n", ("\"" + text + "\"").c_str());
    auto wav = cb->tts(text, ref_audio, ref_sr, /*seed=*/42);
    if (wav.empty()) {
        std::fprintf(stderr, "FAIL: tts returned empty\n");
        return 1;
    }

    float wmn, wmx, wavg, wabs;
    stats(wav, wmn, wmx, wavg, wabs);
    std::printf("\nGenerated wav: %zu samples (~%.2f s @ %d Hz)\n",
                wav.size(),
                static_cast<double>(wav.size()) / cb->output_sample_rate(),
                cb->output_sample_rate());
    std::printf("  min=%+.4f max=%+.4f abs_mean=%.4f\n", wmn, wmx, wabs);

    if (!chatterbox::write_wav_mono(out_wav, wav, cb->output_sample_rate())) {
        std::fprintf(stderr, "FAIL: write_wav %s\n", out_wav.c_str());
        return 1;
    }
    std::printf("\nWrote %s — listen and judge quality.\n", out_wav.c_str());

    int n_fail = 0;
    if (wmn < -1.0f || wmx > 1.0f) {
        std::printf("FAIL: PCM out of [-1, 1]\n");
        ++n_fail;
    }
    if (wabs < 1e-5f) {
        std::printf("FAIL: PCM is essentially silence\n");
        ++n_fail;
    }
    if (n_fail) return 1;
    std::printf("\nPASS\n");
    return 0;
}
