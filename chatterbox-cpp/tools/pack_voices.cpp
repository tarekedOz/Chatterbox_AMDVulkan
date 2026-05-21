// Pack a directory of *.wav voice references into a single GGUF
// sidecar containing pre-computed conditioning blobs for each voice.
//
// Usage:
//   pack_voices <t3.gguf> <ve.gguf> <s3gen.gguf>  \
//                <voices_dir>  <out.gguf>
//
// On a 28-voice set this takes ~25-30 minutes on CPU (most of the
// time is in the 12-mid-block CFM solver and VE LSTM).

#include "chatterbox.h"
#include "voice_pack.h"

#include <cstdio>
#include <string>

int main(int argc, char** argv) {
    if (argc != 6) {
        std::fprintf(stderr,
                     "Usage: %s <t3.gguf> <ve.gguf> <s3gen.gguf>"
                     " <voices_dir> <out.gguf>\n", argv[0]);
        return 2;
    }
    const std::string t3_path    = argv[1];
    const std::string ve_path    = argv[2];
    const std::string s3gen_path = argv[3];
    const std::string voices_dir = argv[4];
    const std::string out_path   = argv[5];

    auto cb = chatterbox::Chatterbox::load(t3_path, ve_path, s3gen_path);
    if (!cb) {
        std::fprintf(stderr, "Chatterbox::load failed\n");
        return 1;
    }
    const int n = chatterbox::VoicePack::pack(*cb, voices_dir, out_path);
    if (n < 0) return 1;
    std::printf("Packed %d voice(s).\n", n);
    return 0;
}
