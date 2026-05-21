#pragma once

// High-level Chatterbox TTS orchestrator. Loads all three GGUFs (T3,
// VE, S3Gen) and exposes a single text + reference-WAV -> PCM method.
//
// Pipeline:
//   1. ref_wav -> resample to 16 kHz and 24 kHz
//   2. From 16 kHz:
//        - VE 40-mel    -> VE LSTM      -> speaker_emb_256 (T3 conditioning)
//        - Kaldi fbank  -> CAMPPlus     -> speaker_emb_192 (S3Gen conditioning)
//        - S3Tok mel    -> S3Encoder    -> FSQ -> prompt_tokens
//   3. From 24 kHz:
//        - S3Gen 80-mel  ->            prompt_feat (T_prompt_mel, 80)
//   4. text -> BPE tokenizer            -> text_tokens
//   5. T3.prefill(speaker_emb_256, prompt_tokens, text_tokens)
//      T3.generate(...)                  -> speech_tokens
//   6. all_tokens = prompt_tokens + speech_tokens
//      FlowEncoder.forward(all_tokens)   -> mu (T_mel = 2 * |all_tokens|, 80)
//      FlowEncoder.affine_speaker(F.normalize(speaker_emb_192))
//                                         -> spks (80,)
//      Build cond: zeros (T_mel, 80); first 2*|prompt_tokens| = prompt_feat
//      Build mask: ones (T_mel,)
//      Generate noise z (deterministic, seeded)
//      CFMDecoder.solve_meanflow(z, mu, mask, spks, cond, T_mel, 2)
//                                         -> output_mel (T_mel, 80)
//      Trim first 2*|prompt_tokens| frames of output_mel.
//   7. HiFTVocoder.predict_f0(trimmed_mel)
//      HiFTVocoder.generate_source(f0, seeded)
//      HiFTVocoder.decode(trimmed_mel, source) -> wav (T_wav,)
//   8. 40 ms cosine fade-in to mask boundary clicks.

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace chatterbox {

class Model;
class Tokenizer;
class T3;
class VE;
class S3Tokenizer;
class S3SpeakerEncoder;
class FlowEncoder;
class CFMDecoder;
class HiFTVocoder;
class KaldiFbankExtractor;
class VEMelExtractor;
class S3GenMelExtractor;

struct ChatterboxConfig {
    int   ref_sr_16k             = 16000;
    int   ref_sr_24k             = 24000;
    int   t3_max_context         = 2048;
    int   t3_max_speech_tokens   = 500;
    int   t3_start_token         = 6561;     // upstream's speech BOS
    int   t3_vocab_limit         = 6561;     // EOS / OOV threshold
    float t3_temperature         = 0.7f;
    int   t3_top_k               = 50;
    float t3_top_p               = 0.95f;
    float t3_repetition_penalty  = 1.5f;
    int   cfm_n_timesteps        = 2;        // meanflow default
    float audio_fade_in_seconds  = 0.04f;    // 40 ms cosine fade
    int   default_seed           = 42;
};

// All four conditioning blobs derived from a reference WAV. Cheap to
// copy and trivially serializable; the VoicePack stores 28 of these.
struct Conditioning {
    std::vector<float>   spk_emb_256;     // VE output (L2-normalized) for T3
    std::vector<float>   spk_emb_192;     // CAMPPlus output (raw) for S3Gen
    std::vector<int32_t> prompt_tokens;   // S3Tokenizer output
    std::vector<float>   prompt_feat;     // (T_mel, 80) S3Gen 24k mel
    int                  prompt_feat_T = 0;
};

class Chatterbox {
public:
    static std::unique_ptr<Chatterbox> load(
        const std::string& t3_gguf_path,
        const std::string& ve_gguf_path,
        const std::string& s3gen_gguf_path,
        const ChatterboxConfig& cfg = {});

    ~Chatterbox();
    Chatterbox(const Chatterbox&) = delete;
    Chatterbox& operator=(const Chatterbox&) = delete;

    // Cache the conditioning derived from a reference WAV. Subsequent
    // synthesize() calls use this conditioning. Returns false on error.
    bool condition(const std::vector<float>& ref_wav, int ref_sr);

    // Extract conditioning from a WAV without caching it. Useful for
    // pre-computing voice packs.
    bool extract_conditioning(const std::vector<float>& ref_wav, int ref_sr,
                                 Conditioning& out) const;

    // Set the cached conditioning to a pre-computed blob (e.g., from a
    // VoicePack).
    void apply_conditioning(const Conditioning& c);

    // Per-request synthesis overrides. Each field falls back to the
    // load-time ChatterboxConfig default when left at its sentinel
    // (negative for the float/int knobs, 0 for seed). This lets a
    // caller tune sampling per request without reloading the engine.
    struct SynthParams {
        uint64_t seed               = 0;     // 0 = config default_seed
        int      max_tokens         = -1;    // <=0 = config t3_max_speech_tokens
        float    temperature        = -1.0f; // <0  = config t3_temperature
        int      top_k              = -1;    // <0  = config t3_top_k
        float    top_p              = -1.0f; // <0  = config t3_top_p
        float    repetition_penalty = -1.0f; // <0  = config t3_repetition_penalty
        int      cfm_timesteps      = -1;    // <=0 = config cfm_n_timesteps
    };

    // Run the full text -> wav pipeline using cached conditioning.
    // Returns 24 kHz mono PCM. Empty on error.
    std::vector<float> synthesize(const std::string& text,
                                     const SynthParams& params);

    // Backwards-compatible overload: seed + max_speech_tokens only.
    std::vector<float> synthesize(const std::string& text,
                                     uint64_t seed = 0,
                                     int max_speech_tokens = -1);

    // Convenience: condition + synthesize. Equivalent to:
    //   condition(ref_wav, ref_sr); return synthesize(text, seed, max);
    std::vector<float> tts(const std::string& text,
                              const std::vector<float>& ref_wav, int ref_sr,
                              uint64_t seed = 0,
                              int max_speech_tokens = -1);

    // Direct accessors for inspection / smoke tests.
    int  prompt_tokens_count() const;
    int  prompt_feat_T() const;
    bool is_conditioned() const { return conditioned_; }
    const ChatterboxConfig& config() const { return cfg_; }
    int  output_sample_rate() const { return cfg_.ref_sr_24k; }

private:
    Chatterbox() = default;

    ChatterboxConfig cfg_;

    // Models (own the GGUF buffers)
    std::unique_ptr<Model> t3_model_;
    std::unique_ptr<Model> ve_model_;
    std::unique_ptr<Model> s3gen_model_;

    // Component instances
    std::unique_ptr<Tokenizer>            text_tok_;
    std::unique_ptr<T3>                   t3_;
    std::unique_ptr<VE>                   ve_;
    std::unique_ptr<S3Tokenizer>          s3_tok_;
    std::unique_ptr<S3SpeakerEncoder>     camplus_;
    std::unique_ptr<FlowEncoder>          flow_;
    std::unique_ptr<CFMDecoder>           cfm_;
    std::unique_ptr<HiFTVocoder>          hift_;

    // Host-side extractors
    std::unique_ptr<KaldiFbankExtractor>  kaldi_;
    std::unique_ptr<VEMelExtractor>       ve_mel_;
    std::unique_ptr<S3GenMelExtractor>    s3gen_mel_;

    // Cached conditioning
    std::vector<float>   spk_emb_256_;     // L2-normalized (for T3 prefill)
    std::vector<float>   spk_emb_192_;     // raw (will be L2-normed in synth)
    std::vector<float>   prompt_feat_;     // (T_mel_24k, 80) row-major
    int                  prompt_feat_T_ = 0;
    std::vector<int32_t> prompt_tokens_;
    bool                 conditioned_   = false;
};

}  // namespace chatterbox
