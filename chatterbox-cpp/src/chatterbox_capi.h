#pragma once

// Plain C ABI wrapper around chatterbox::Chatterbox + VoicePack. The
// Rust server (chatterbox-server crate) links this and goes through
// extern "C" declarations.
//
// Lifetime: one engine context per process. Created with
// chatterbox_init, freed with chatterbox_free. Thread safety: the
// engine is not thread-safe; serialize synthesize calls externally.

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

typedef struct chatterbox_ctx chatterbox_ctx_t;

// Initialize the engine. Returns NULL on error.
chatterbox_ctx_t* chatterbox_init(const char* t3_gguf_path,
                                     const char* ve_gguf_path,
                                     const char* s3gen_gguf_path);

void chatterbox_free(chatterbox_ctx_t* ctx);

// Load a voices.gguf voice pack. Returns 0 on success, negative on
// error.
int chatterbox_load_voices(chatterbox_ctx_t* ctx,
                              const char* voice_pack_path);

// Number of voices in the loaded pack.
int chatterbox_voice_count(const chatterbox_ctx_t* ctx);

// Copy the i-th voice's name into `out` (max `max_len` bytes including
// NUL). Returns the number of bytes written (excluding NUL), or
// negative on error.
int chatterbox_voice_name(const chatterbox_ctx_t* ctx, int index,
                              char* out, int max_len);

// Set the active conditioning from a named voice in the pack.
// Returns 0 on success, negative on error (e.g. voice not found).
int chatterbox_set_voice(chatterbox_ctx_t* ctx, const char* voice_name);

// Clone a voice from raw reference audio. `samples` is mono fp32 PCM at
// `sr` Hz (any rate; the engine resamples internally). Caches the
// derived conditioning so subsequent chatterbox_synthesize[_ex] calls
// use this voice, until set_voice / condition_pcm is called again.
// Returns 0 on success, negative on error.
int chatterbox_condition_pcm(chatterbox_ctx_t* ctx,
                                 const float* samples, int n_samples, int sr);

// Per-request generation parameters. Each field uses a sentinel that
// means "use the engine's load-time default": negative for the float
// and int knobs, 0 for seed/max_tokens. NOTE: a zero-initialized struct
// leaves the float fields at 0.0, which is NOT the default sentinel —
// callers that want defaults must set the float fields negative (the
// Rust wrapper and chatterbox_synthesize() below do this).
typedef struct {
    uint64_t seed;               // 0  = default seed
    int      max_tokens;         // <=0 = default (t3_max_speech_tokens)
    float    temperature;        // <0 = default
    int      top_k;              // <0 = default
    float    top_p;              // <0 = default
    float    repetition_penalty; // <0 = default
    int      cfm_timesteps;      // <=0 = default (meanflow steps)
    // Accepted but IGNORED: the Turbo model does not support these
    // (upstream Turbo ignores them too). Kept for ABI stability / a
    // possible future non-Turbo target. See
    // docs/exaggeration-cfg-investigation.md.
    float    exaggeration;       // ignored
    float    cfg_weight;         // ignored
} chatterbox_gen_params_t;

// Synthesize. `out_pcm` is a caller-provided buffer of `max_samples`
// fp32 samples. Returns the number of samples written (always
// monophonic), or negative on error. The output sample rate is
// written to *out_sample_rate (24000 for chatterbox).
//
// `seed=0` uses the default seed. `max_tokens=0` uses the default
// (500). Internally caps audio at ~16 seconds (text-dependent).
int chatterbox_synthesize(chatterbox_ctx_t* ctx,
                              const char* text,
                              uint64_t seed,
                              int max_tokens,
                              float* out_pcm, int max_samples,
                              int* out_sample_rate);

// Synthesize with full per-request params. `params` may be NULL (all
// defaults). Same return/buffer contract as chatterbox_synthesize.
int chatterbox_synthesize_ex(chatterbox_ctx_t* ctx,
                                 const char* text,
                                 const chatterbox_gen_params_t* params,
                                 float* out_pcm, int max_samples,
                                 int* out_sample_rate);

// Last error message in TLS (one slot, NUL-terminated). Returns NULL
// if no error has been set yet. The returned pointer is owned by the
// library and valid until the next library call.
const char* chatterbox_last_error(void);

#ifdef __cplusplus
}
#endif
