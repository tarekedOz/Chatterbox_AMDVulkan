#pragma once

// A bundle of pre-computed conditioning blobs for N reference voices,
// serialized as a GGUF sidecar file.
//
// File schema (arch = "chatterbox_voices"):
//
//   KV:
//     general.architecture          = "chatterbox_voices"
//     general.name                  = "chatterbox_voices_v1"
//     voices.count                  = uint32 (N)
//     voices.names                  = string array of length N
//
//   Tensors (per voice i in 0..N-1):
//     voice.{i}.spk_emb_256          F32 (256,)        from VoiceEncoder
//     voice.{i}.spk_emb_192          F32 (192,)        from CAMPPlus
//     voice.{i}.prompt_tokens        I32 (T_tok,)      from S3Tokenizer
//     voice.{i}.prompt_feat          F32 (T_mel, 80)   from S3Gen mel
//
// Stored as fp32 throughout — the cost is tiny (~5 MB total for 28
// voices) and avoids the fp16 conversion losses we already pay in
// the model weights.
//
// Pack workflow:
//   VoicePack::pack(chatterbox, "voices_dir/", "voices.gguf")
//     -> reads N wav files from voices_dir,
//        runs Chatterbox::extract_conditioning on each,
//        writes the resulting bundle to voices.gguf.
//
// Read workflow:
//   auto pack = VoicePack::load("voices.gguf");
//   auto names = pack->names();
//   Conditioning c;
//   if (!pack->get_conditioning("Abigail", c)) ...;
//   chatterbox.apply_conditioning(c);

#include "chatterbox.h"

#include <memory>
#include <string>
#include <vector>

namespace chatterbox {

class VoicePack {
public:
    // ---- Read ----
    static std::unique_ptr<VoicePack> load(const std::string& path);

    // Number of voices in the pack.
    size_t size() const { return names_.size(); }

    // All voice names (in stored order).
    const std::vector<std::string>& names() const { return names_; }

    // Look up a voice's conditioning by name. Returns false if not found.
    bool get_conditioning(const std::string& name, Conditioning& out) const;

    // ---- Pack ----
    //
    // Iterate `*.wav` files in `voice_dir`, extract conditioning via
    // chatterbox, and write a GGUF bundle to `out_path`. Voice name is
    // the filename stem (e.g., "Abigail.wav" -> "Abigail"). Returns
    // the number of voices written, or -1 on error.
    //
    // `chatterbox` is borrowed for the duration of this call.
    static int pack(Chatterbox& cb,
                      const std::string& voice_dir,
                      const std::string& out_path,
                      bool verbose = true);

private:
    VoicePack() = default;

    std::vector<std::string>   names_;
    std::vector<Conditioning>  conds_;
};

}  // namespace chatterbox
