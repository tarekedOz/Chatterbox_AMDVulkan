// Voice pack round-trip test: load voices.gguf, look up a named voice,
// apply its conditioning, run synthesize.

#include "chatterbox.h"
#include "voice_pack.h"
#include "wav_io.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 6) {
        std::fprintf(stderr,
                     "Usage: %s <t3.gguf> <ve.gguf> <s3gen.gguf>\n"
                     "          <voices.gguf> <voice_name>\n", argv[0]);
        return 2;
    }
    const std::string t3_path     = argv[1];
    const std::string ve_path     = argv[2];
    const std::string s3gen_path  = argv[3];
    const std::string voices_path = argv[4];
    const std::string voice_name  = argv[5];

    std::printf("Loading voice pack: %s\n", voices_path.c_str());
    auto pack = chatterbox::VoicePack::load(voices_path);
    if (!pack) return 1;
    std::printf("  %zu voices:\n", pack->size());
    for (const auto& n : pack->names()) std::printf("    - %s\n", n.c_str());

    chatterbox::Conditioning c;
    if (!pack->get_conditioning(voice_name, c)) {
        std::fprintf(stderr, "Voice '%s' not in pack.\n", voice_name.c_str());
        return 1;
    }
    std::printf("Voice '%s': spk_emb_256=%zu spk_emb_192=%zu "
                "prompt_tokens=%zu prompt_feat=(%d, 80)\n",
                voice_name.c_str(),
                c.spk_emb_256.size(), c.spk_emb_192.size(),
                c.prompt_tokens.size(), c.prompt_feat_T);

    chatterbox::ChatterboxConfig cfg;
    cfg.t3_max_speech_tokens = 50;
    auto cb = chatterbox::Chatterbox::load(t3_path, ve_path, s3gen_path, cfg);
    if (!cb) return 1;
    cb->apply_conditioning(c);

    const std::string text = "The quick brown fox jumps over the lazy dog.";
    std::printf("\nSynthesizing: %s\n", ("\"" + text + "\"").c_str());
    auto wav = cb->synthesize(text, /*seed=*/42);
    if (wav.empty()) { std::fprintf(stderr, "synthesize failed\n"); return 1; }

    float mn = wav[0], mx = wav[0]; double s = 0.0, as = 0.0;
    for (float v : wav) { if (v<mn) mn=v; if (v>mx) mx=v; s+=v; as+=std::abs(v); }
    std::printf("Generated %zu samples (%.2f s)  min=%+.3f max=%+.3f abs_mean=%.3f\n",
                wav.size(),
                static_cast<double>(wav.size()) / cb->output_sample_rate(),
                mn, mx, static_cast<float>(as / wav.size()));

    if (mn < -1.0f || mx > 1.0f) { std::printf("FAIL: clipped\n"); return 1; }
    if (as / wav.size() < 1e-5) { std::printf("FAIL: silence\n"); return 1; }

    const std::string out_wav = "voice_pack_test_out.wav";
    if (chatterbox::write_wav_mono(out_wav, wav, cb->output_sample_rate())) {
        std::printf("Wrote %s\n", out_wav.c_str());
    }
    std::printf("\nPASS\n");
    return 0;
}
