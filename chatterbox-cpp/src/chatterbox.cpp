#include "chatterbox.h"

#include "cfm_decoder.h"
#include "flow_encoder.h"
#include "hift_vocoder.h"
#include "kaldi_fbank.h"
#include "mel.h"
#include "model.h"
#include "resample.h"
#include "s3enc.h"
#include "s3gen_mel.h"
#include "s3spk.h"
#include "s3tok.h"
#include "sampler.h"
#include "t3.h"
#include "tokenizer.h"
#include "ve.h"
#include "ve_mel.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {
// Set CHATTERBOX_PROFILE=1 to print per-stage wall-clock timings.
bool profile_enabled() {
    const char* v = std::getenv("CHATTERBOX_PROFILE");
    return v && v[0] && v[0] != '0';
}
struct StageTimer {
    const char* name;
    std::chrono::steady_clock::time_point t0;
    bool on;
    StageTimer(const char* n)
        : name(n), t0(std::chrono::steady_clock::now()), on(profile_enabled()) {}
    ~StageTimer() {
        if (!on) return;
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::printf("[profile] %-22s %10.2f ms\n", name, ms);
    }
};
}

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace chatterbox {

namespace {

// L2-normalize a vector in-place (no-op for zero vectors).
void l2_normalize(std::vector<float>& v) {
    double s = 0.0;
    for (float x : v) s += static_cast<double>(x) * x;
    const float norm = static_cast<float>(std::sqrt(s));
    if (norm > 1e-12f) {
        const float inv = 1.0f / norm;
        for (auto& x : v) x *= inv;
    }
}

}  // namespace


// ----------------------------------------------------------------------------
// Loader
// ----------------------------------------------------------------------------

Chatterbox::~Chatterbox() = default;

std::unique_ptr<Chatterbox> Chatterbox::load(
        const std::string& t3_path,
        const std::string& ve_path,
        const std::string& s3gen_path,
        const ChatterboxConfig& cfg) {
    auto cb = std::unique_ptr<Chatterbox>(new Chatterbox());
    cb->cfg_ = cfg;

    cb->t3_model_    = load_model(t3_path);
    cb->ve_model_    = load_model(ve_path);
    cb->s3gen_model_ = load_model(s3gen_path);
    if (!cb->t3_model_ || !cb->ve_model_ || !cb->s3gen_model_) {
        std::fprintf(stderr, "Chatterbox::load: model load failed\n");
        return nullptr;
    }

    cb->text_tok_ = Tokenizer::load(cb->t3_model_->gguf());
    cb->t3_       = T3::load(cb->t3_model_.get(), cfg.t3_max_context);
    cb->ve_       = VE::load(cb->ve_model_.get());
    cb->s3_tok_   = S3Tokenizer::load(cb->s3gen_model_.get());
    cb->camplus_  = S3SpeakerEncoder::load(cb->s3gen_model_.get());
    cb->flow_     = FlowEncoder::load(cb->s3gen_model_.get());
    cb->cfm_      = CFMDecoder::load(cb->s3gen_model_.get());
    cb->hift_     = HiFTVocoder::load(cb->s3gen_model_.get());
    if (!cb->text_tok_ || !cb->t3_ || !cb->ve_ || !cb->s3_tok_ ||
        !cb->camplus_ || !cb->flow_ || !cb->cfm_ || !cb->hift_) {
        std::fprintf(stderr, "Chatterbox::load: component init failed\n");
        return nullptr;
    }

    cb->kaldi_     = KaldiFbankExtractor::create();
    cb->ve_mel_    = VEMelExtractor::create();
    cb->s3gen_mel_ = S3GenMelExtractor::create();

    return cb;
}


// ----------------------------------------------------------------------------
// Conditioning
// ----------------------------------------------------------------------------

bool Chatterbox::extract_conditioning(const std::vector<float>& ref_wav,
                                          int ref_sr,
                                          Conditioning& out) const {
    if (ref_wav.empty()) return false;

    // 1. Resample to 16 kHz and 24 kHz.
    const std::vector<float>* wav_16k_p = &ref_wav;
    std::vector<float> wav_16k_buf;
    const std::vector<float>* wav_24k_p = &ref_wav;
    std::vector<float> wav_24k_buf;
    {
        StageTimer _t("cond_resample");
        if (ref_sr != cfg_.ref_sr_16k) {
            wav_16k_buf = resample_audio(ref_wav, ref_sr, cfg_.ref_sr_16k);
            wav_16k_p = &wav_16k_buf;
        }
        if (ref_sr != cfg_.ref_sr_24k) {
            wav_24k_buf = resample_audio(ref_wav, ref_sr, cfg_.ref_sr_24k);
            wav_24k_p = &wav_24k_buf;
        }
    }
    const auto& wav_16k = *wav_16k_p;
    const auto& wav_24k = *wav_24k_p;
    std::printf("Chatterbox::condition: ref %.2f s, 16k=%zu samples, "
                "24k=%zu samples\n",
                static_cast<double>(ref_wav.size()) / ref_sr,
                wav_16k.size(), wav_24k.size());

    // 2. VE 40-mel + VE LSTM -> 256-d L2-normalized speaker embedding.
    // VE's LSTM unrolls one cell per mel frame * 3 layers. We cap the
    // audio length to keep the graph reasonable. Upstream's
    // VoiceEncoder.embeds_from_wavs averages 800ms chunks; for our
    // simpler path we just take the leading 6 seconds of audio (or
    // less if the reference is shorter). Speaker identity is well-
    // captured in even 1-2 seconds of speech.
    constexpr int VE_MAX_SAMPLES = 6 * 16000;
    std::vector<float> wav_ve(wav_16k.begin(),
        wav_16k.begin() + std::min<size_t>(wav_16k.size(), VE_MAX_SAMPLES));
    int ve_T = 0, ve_M = 0;
    {
        StageTimer _t("cond_ve_mel");
        auto ve_mel = ve_mel_->compute(wav_ve, ve_T, ve_M);
        StageTimer _t2("cond_ve_lstm");
        out.spk_emb_256 = ve_->forward(ve_mel, ve_T);
    }
    if (out.spk_emb_256.empty()) {
        std::fprintf(stderr, "extract_conditioning: VE forward failed\n");
        return false;
    }

    // 3. Kaldi fbank + CAMPPlus -> 192-d (unnormalized) speaker embedding.
    int fb_T = 0, fb_M = 0;
    {
        StageTimer _t("cond_kaldi_fbank");
        auto fbank = kaldi_->extract(wav_16k, fb_T, fb_M);
        StageTimer _t2("cond_camplus");
        out.spk_emb_192 = camplus_->forward(fbank, fb_T, fb_M);
    }
    if (out.spk_emb_192.empty()) {
        std::fprintf(stderr, "extract_conditioning: CAMPPlus forward failed\n");
        return false;
    }

    // 4. S3 tokenizer -> prompt speech tokens.
    {
        StageTimer _t("cond_s3_tokenizer");
        out.prompt_tokens = s3_tok_->encode(wav_16k);
    }
    if (out.prompt_tokens.empty()) {
        std::fprintf(stderr,
                     "extract_conditioning: S3Tokenizer.encode returned empty\n");
        return false;
    }

    // 5. S3Gen 24 kHz 80-mel -> prompt_feat (T_mel, 80).
    int pm_T = 0, pm_M = 0;
    StageTimer _t_mel("cond_s3gen_mel");
    auto prompt_feat_cf = s3gen_mel_->log_mel(wav_24k, pm_M, pm_T);
    out.prompt_feat.assign(static_cast<size_t>(pm_T) * pm_M, 0.0f);
    for (int t = 0; t < pm_T; ++t) {
        for (int m = 0; m < pm_M; ++m) {
            out.prompt_feat[static_cast<size_t>(t) * pm_M + m] =
                prompt_feat_cf[static_cast<size_t>(m) * pm_T + t];
        }
    }
    out.prompt_feat_T = pm_T;
    std::printf("  speaker_emb_256: %zu-d  speaker_emb_192: %zu-d\n",
                out.spk_emb_256.size(), out.spk_emb_192.size());
    std::printf("  prompt_tokens: %zu  prompt_feat: (%d, %d)\n",
                out.prompt_tokens.size(), pm_T, pm_M);

    // Trim prompt_feat so that prompt_feat_T <= 2*N_prompt.
    const int expected_mel_len = 2 * static_cast<int>(out.prompt_tokens.size());
    if (out.prompt_feat_T > expected_mel_len) {
        out.prompt_feat.resize(static_cast<size_t>(expected_mel_len) * pm_M);
        out.prompt_feat_T = expected_mel_len;
        std::printf("  prompt_feat trimmed to T=%d to match 2*N_prompt\n",
                    out.prompt_feat_T);
    }
    return true;
}

void Chatterbox::apply_conditioning(const Conditioning& c) {
    spk_emb_256_   = c.spk_emb_256;
    spk_emb_192_   = c.spk_emb_192;
    prompt_tokens_ = c.prompt_tokens;
    prompt_feat_   = c.prompt_feat;
    prompt_feat_T_ = c.prompt_feat_T;
    conditioned_   = !spk_emb_256_.empty()
                      && !spk_emb_192_.empty()
                      && !prompt_tokens_.empty();
}

bool Chatterbox::condition(const std::vector<float>& ref_wav, int ref_sr) {
    conditioned_ = false;
    Conditioning c;
    if (!extract_conditioning(ref_wav, ref_sr, c)) return false;
    apply_conditioning(c);
    return conditioned_;
}

int Chatterbox::prompt_tokens_count() const {
    return static_cast<int>(prompt_tokens_.size());
}

int Chatterbox::prompt_feat_T() const { return prompt_feat_T_; }


// ----------------------------------------------------------------------------
// Synthesize
// ----------------------------------------------------------------------------

std::vector<float> Chatterbox::synthesize(const std::string& text,
                                              uint64_t seed,
                                              int max_speech_tokens) {
    SynthParams p;
    p.seed       = seed;
    p.max_tokens = max_speech_tokens;
    return synthesize(text, p);
}

std::vector<float> Chatterbox::synthesize(const std::string& text,
                                              const SynthParams& params) {
    if (!conditioned_) {
        std::fprintf(stderr,
                     "Chatterbox::synthesize: must call condition() first\n");
        return {};
    }
    // Resolve per-request overrides against the load-time config defaults.
    const uint64_t seed = params.seed != 0 ? params.seed
                                            : static_cast<uint64_t>(cfg_.default_seed);
    const int max_speech_tokens = params.max_tokens > 0
        ? params.max_tokens : cfg_.t3_max_speech_tokens;
    const float temperature = params.temperature >= 0.0f
        ? params.temperature : cfg_.t3_temperature;
    const int top_k = params.top_k >= 0 ? params.top_k : cfg_.t3_top_k;
    const float top_p = params.top_p >= 0.0f ? params.top_p : cfg_.t3_top_p;
    const float repetition_penalty = params.repetition_penalty >= 0.0f
        ? params.repetition_penalty : cfg_.t3_repetition_penalty;
    const int cfm_timesteps = params.cfm_timesteps > 0
        ? params.cfm_timesteps : cfg_.cfm_n_timesteps;

    // 1. Tokenize text.
    auto text_tokens = text_tok_->encode(text);
    std::printf("Chatterbox::synthesize: text -> %zu tokens\n",
                text_tokens.size());

    // 2. T3 prefill + generate.
    std::vector<float> pre_logits;
    {
        StageTimer _t("t3_prefill");
        pre_logits = t3_->prefill(spk_emb_256_, prompt_tokens_, text_tokens);
    }
    if (pre_logits.empty()) {
        std::fprintf(stderr, "synthesize: T3.prefill failed\n");
        return {};
    }
    T3::GenParams gp;
    gp.sampling.seed              = static_cast<uint32_t>(seed);
    gp.sampling.temperature       = temperature;
    gp.sampling.top_k             = top_k;
    gp.sampling.top_p             = top_p;
    gp.sampling.repetition_penalty = repetition_penalty;
    gp.max_tokens   = max_speech_tokens;
    gp.start_token  = cfg_.t3_start_token;
    gp.vocab_limit  = cfg_.t3_vocab_limit;
    std::vector<int32_t> gen_speech;
    {
        StageTimer _t("t3_generate");
        gen_speech = t3_->generate(gp);
    }
    std::printf("  T3 generated %zu speech tokens\n", gen_speech.size());
    if (gen_speech.empty()) {
        std::fprintf(stderr, "synthesize: T3.generate produced 0 tokens\n");
        return {};
    }

    // 3. Concat prompt + generated tokens. Cast prompt to int32.
    std::vector<int32_t> all_tokens;
    all_tokens.reserve(prompt_tokens_.size() + gen_speech.size());
    for (int32_t t : prompt_tokens_) all_tokens.push_back(t);
    for (int32_t t : gen_speech)     all_tokens.push_back(t);

    // 4. Flow encoder. Returns (T_mel, 80) row-major where T_mel = 2 * |all|.
    int mu_T = 0, mu_d = 0;
    std::vector<float> mu_TC;
    {
        StageTimer _t("flow_encoder");
        mu_TC = flow_->forward(all_tokens, mu_T, mu_d);
    }
    if (mu_TC.empty()) {
        std::fprintf(stderr, "synthesize: FlowEncoder.forward failed\n");
        return {};
    }
    std::printf("  Flow encoder: mu (%d, %d)\n", mu_T, mu_d);

    // 5. spks via affine_speaker (with L2-normalized 192-d input).
    std::vector<float> spk192 = spk_emb_192_;
    l2_normalize(spk192);
    std::vector<float> spks;
    {
        StageTimer _t("flow_affine_speaker");
        spks = flow_->affine_speaker(spk192);
    }
    if (spks.empty()) {
        std::fprintf(stderr, "synthesize: affine_speaker failed\n");
        return {};
    }

    // 6. cond: (T_mel, 80) row-major with first prompt_feat_T_ rows = prompt_feat.
    std::vector<float> cond(static_cast<size_t>(mu_T) * mu_d, 0.0f);
    if (prompt_feat_T_ > 0 && prompt_feat_T_ <= mu_T) {
        std::memcpy(cond.data(), prompt_feat_.data(),
                     static_cast<size_t>(prompt_feat_T_) * mu_d * sizeof(float));
    }

    // 7. mask: (T_mel,) all ones.
    std::vector<float> mask(mu_T, 1.0f);

    // 8. Generate noise z: (T_mel, 80) row-major, seeded.
    std::mt19937 rng(static_cast<uint64_t>(seed) + 1);
    std::normal_distribution<float> ndist(0.0f, 1.0f);
    std::vector<float> z(static_cast<size_t>(mu_T) * mu_d);
    for (auto& v : z) v = ndist(rng);
    // Set z's prompt portion to the prompt_feat (matches noised_mels=None
    // behaviour where the model conditions on real prompt mels).
    if (prompt_feat_T_ > 0 && prompt_feat_T_ <= mu_T) {
        // Actually upstream zeroes out the prompt portion of noise so the
        // model can use the cond rather than a noisy version. But the
        // default basic_euler path just uses fresh noise everywhere. Match
        // that.
    }

    // 9. CFM solve.
    std::vector<float> out_mel;
    {
        StageTimer _t("cfm_solve");
        out_mel = cfm_->solve_meanflow(z, mu_TC, mask, spks, cond,
                                          mu_T, cfm_timesteps);
    }
    if (out_mel.empty()) {
        std::fprintf(stderr, "synthesize: CFM solver failed\n");
        return {};
    }
    std::printf("  CFM solver: out_mel %d frames\n", mu_T);

    // 10. Trim prompt portion: keep only [prompt_feat_T_:] rows.
    const int gen_mel_T = mu_T - prompt_feat_T_;
    std::vector<float> gen_mel(
        out_mel.begin() + static_cast<size_t>(prompt_feat_T_) * mu_d,
        out_mel.begin() + static_cast<size_t>(prompt_feat_T_ + gen_mel_T) * mu_d);
    std::printf("  Trimmed mel: %d frames -> wav at %d Hz "
                "(expected %d samples)\n",
                gen_mel_T, cfg_.ref_sr_24k,
                gen_mel_T * hift_->config().upsample_factor);

    // 11. F0 prediction + NSF source.
    std::vector<float> f0;
    {
        StageTimer _t("hift_predict_f0");
        f0 = hift_->predict_f0(gen_mel, gen_mel_T);
    }
    if (f0.empty()) {
        std::fprintf(stderr, "synthesize: F0 predictor failed\n");
        return {};
    }
    std::vector<float> source;
    {
        StageTimer _t("hift_gen_source");
        source = hift_->generate_source(f0, /*phase=*/{}, /*noise=*/{},
                                            /*seed=*/seed + 2);
    }
    if (source.empty()) {
        std::fprintf(stderr, "synthesize: generate_source failed\n");
        return {};
    }
    const int T_wav = static_cast<int>(source.size());

    // 12. HiFT decode.
    std::vector<float> wav;
    {
        StageTimer _t("hift_decode");
        wav = hift_->decode(gen_mel, gen_mel_T, source, T_wav);
    }
    if (wav.empty()) {
        std::fprintf(stderr, "synthesize: HiFT decode failed\n");
        return {};
    }
    std::printf("  HiFT decode: %zu samples (~%.2f s @ %d Hz)\n",
                wav.size(),
                static_cast<double>(wav.size()) / cfg_.ref_sr_24k,
                cfg_.ref_sr_24k);

    // 13. 40 ms cosine fade-in.
    const int fade_n = std::min<int>(
        static_cast<int>(wav.size()),
        static_cast<int>(cfg_.audio_fade_in_seconds * cfg_.ref_sr_24k));
    for (int i = 0; i < fade_n; ++i) {
        const float w = 0.5f - 0.5f *
            std::cos(static_cast<float>(M_PI) * i / fade_n);
        wav[i] *= w;
    }

    return wav;
}

std::vector<float> Chatterbox::tts(const std::string& text,
                                       const std::vector<float>& ref_wav,
                                       int ref_sr,
                                       uint64_t seed,
                                       int max_speech_tokens) {
    {
        StageTimer _t("condition_total");
        if (!condition(ref_wav, ref_sr)) return {};
    }
    StageTimer _t("synthesize_total");
    return synthesize(text, seed, max_speech_tokens);
}

}  // namespace chatterbox
