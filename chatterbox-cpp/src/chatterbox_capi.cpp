#include "chatterbox_capi.h"

#include "chatterbox.h"
#include "voice_pack.h"

#include <algorithm>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace {

thread_local std::string g_last_error;

void set_err(const std::string& s) {
    g_last_error = s;
}

}  // namespace

struct chatterbox_ctx {
    std::unique_ptr<chatterbox::Chatterbox>   engine;
    std::unique_ptr<chatterbox::VoicePack>    voices;
};


extern "C" {

chatterbox_ctx_t* chatterbox_init(const char* t3_path,
                                     const char* ve_path,
                                     const char* s3gen_path) {
    if (!t3_path || !ve_path || !s3gen_path) {
        set_err("chatterbox_init: null path argument");
        return nullptr;
    }
    auto ctx = std::make_unique<chatterbox_ctx>();
    ctx->engine = chatterbox::Chatterbox::load(t3_path, ve_path, s3gen_path);
    if (!ctx->engine) {
        set_err("chatterbox_init: Chatterbox::load failed");
        return nullptr;
    }
    return ctx.release();
}

void chatterbox_free(chatterbox_ctx_t* ctx) {
    delete ctx;
}

int chatterbox_load_voices(chatterbox_ctx_t* ctx, const char* path) {
    if (!ctx || !path) return -1;
    ctx->voices = chatterbox::VoicePack::load(path);
    if (!ctx->voices) {
        set_err("chatterbox_load_voices: VoicePack::load failed");
        return -1;
    }
    return 0;
}

int chatterbox_voice_count(const chatterbox_ctx_t* ctx) {
    if (!ctx || !ctx->voices) return 0;
    return static_cast<int>(ctx->voices->size());
}

int chatterbox_voice_name(const chatterbox_ctx_t* ctx, int index,
                              char* out, int max_len) {
    if (!ctx || !ctx->voices || index < 0 || !out || max_len <= 0) {
        return -1;
    }
    const auto& names = ctx->voices->names();
    if (index >= static_cast<int>(names.size())) return -1;
    const std::string& n = names[index];
    const int copy = std::min<int>(static_cast<int>(n.size()), max_len - 1);
    std::memcpy(out, n.data(), copy);
    out[copy] = '\0';
    return copy;
}

int chatterbox_set_voice(chatterbox_ctx_t* ctx, const char* name) {
    if (!ctx || !ctx->engine) return -1;
    if (!ctx->voices) {
        set_err("chatterbox_set_voice: no voice pack loaded");
        return -1;
    }
    chatterbox::Conditioning c;
    if (!ctx->voices->get_conditioning(name, c)) {
        set_err(std::string("chatterbox_set_voice: unknown voice: ") + name);
        return -1;
    }
    ctx->engine->apply_conditioning(c);
    return 0;
}

int chatterbox_condition_pcm(chatterbox_ctx_t* ctx,
                                 const float* samples, int n_samples, int sr) {
    if (!ctx || !ctx->engine || !samples || n_samples <= 0 || sr <= 0) {
        set_err("chatterbox_condition_pcm: invalid argument");
        return -1;
    }
    std::vector<float> ref(samples, samples + n_samples);
    if (!ctx->engine->condition(ref, sr)) {
        set_err("chatterbox_condition_pcm: conditioning failed");
        return -1;
    }
    return 0;
}

int chatterbox_synthesize_ex(chatterbox_ctx_t* ctx,
                                 const char* text,
                                 const chatterbox_gen_params_t* params,
                                 float* out_pcm, int max_samples,
                                 int* out_sample_rate) {
    if (!ctx || !ctx->engine || !text || !out_pcm || max_samples <= 0) {
        return -1;
    }
    chatterbox::Chatterbox::SynthParams sp;  // all sentinels = defaults
    if (params) {
        sp.seed               = params->seed;
        sp.max_tokens         = params->max_tokens;
        sp.temperature        = params->temperature;
        sp.top_k              = params->top_k;
        sp.top_p              = params->top_p;
        sp.repetition_penalty = params->repetition_penalty;
        sp.cfm_timesteps      = params->cfm_timesteps;
        // exaggeration / cfg_weight: ignored — unsupported by Turbo
        // (see docs/exaggeration-cfg-investigation.md).
    }
    auto wav = ctx->engine->synthesize(text, sp);
    if (wav.empty()) {
        set_err("chatterbox_synthesize: empty output");
        return -1;
    }
    const int n = std::min<int>(static_cast<int>(wav.size()), max_samples);
    std::memcpy(out_pcm, wav.data(), n * sizeof(float));
    if (out_sample_rate) *out_sample_rate = ctx->engine->output_sample_rate();
    return n;
}

int chatterbox_synthesize(chatterbox_ctx_t* ctx,
                              const char* text,
                              uint64_t seed,
                              int max_tokens,
                              float* out_pcm, int max_samples,
                              int* out_sample_rate) {
    chatterbox_gen_params_t p;
    p.seed               = seed;
    p.max_tokens         = max_tokens;
    p.temperature        = -1.0f;   // sentinels -> engine defaults
    p.top_k              = -1;
    p.top_p              = -1.0f;
    p.repetition_penalty = -1.0f;
    p.cfm_timesteps      = -1;
    p.exaggeration       = -1.0f;
    p.cfg_weight         = -1.0f;
    return chatterbox_synthesize_ex(ctx, text, &p, out_pcm, max_samples,
                                    out_sample_rate);
}

const char* chatterbox_last_error(void) {
    return g_last_error.empty() ? nullptr : g_last_error.c_str();
}

}  // extern "C"
