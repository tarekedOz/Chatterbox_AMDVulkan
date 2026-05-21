#include "t3.h"
#include "backend.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace chatterbox {

namespace {

constexpr size_t GRAPH_MAX_NODES = 16384;

// Per-step timing accumulators for T3 forward_impl. Enable with
// CHATTERBOX_T3_PROFILE=1. Reset and printed inside T3::generate.
struct T3StepStats {
    double build_ms      = 0.0;
    double sched_ms      = 0.0;
    double set_inputs_ms = 0.0;
    double compute_ms    = 0.0;
    double readback_ms   = 0.0;
    int    n_steps       = 0;
};
T3StepStats g_t3_stats;
bool t3_profile_enabled() {
    const char* v = std::getenv("CHATTERBOX_T3_PROFILE");
    return v && v[0] && v[0] != '0';
}
double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
}

uint32_t read_u32(gguf_context* gguf, const char* key, uint32_t fallback) {
    int64_t id = gguf_find_key(gguf, key);
    return id < 0 ? fallback : gguf_get_val_u32(gguf, id);
}

float read_f32(gguf_context* gguf, const char* key, float fallback) {
    int64_t id = gguf_find_key(gguf, key);
    return id < 0 ? fallback : gguf_get_val_f32(gguf, id);
}

}  // namespace

T3::~T3() {
    if (sched_)        ggml_backend_sched_free(sched_);
    if (tw_buffer_)    ggml_backend_buffer_free(tw_buffer_);
    if (tw_ctx_)       ggml_free(tw_ctx_);
    if (cache_buffer_) ggml_backend_buffer_free(cache_buffer_);
    if (cache_ctx_)    ggml_free(cache_ctx_);
    // backend_ is a process-wide singleton (see backend.h); do NOT free.
}

std::unique_ptr<T3> T3::load(Model* model, int max_context) {
    if (!model) return nullptr;
    if (model->arch != "chatterbox_t3") {
        std::fprintf(stderr,
                     "T3::load: expected arch chatterbox_t3, got %s\n",
                     model->arch.c_str());
        return nullptr;
    }
    if (max_context <= 0) {
        std::fprintf(stderr, "T3::load: max_context must be > 0\n");
        return nullptr;
    }

    auto t3 = std::unique_ptr<T3>(new T3());
    t3->model_ = model;
    t3->max_context_ = max_context;

    gguf_context* gguf = model->gguf();
    t3->cfg_.n_layers        = (int32_t)read_u32(gguf, "chatterbox_t3.block_count",              24);
    t3->cfg_.n_heads         = (int32_t)read_u32(gguf, "chatterbox_t3.attention.head_count",     16);
    t3->cfg_.embed_dim       = (int32_t)read_u32(gguf, "chatterbox_t3.embedding_length",       1024);
    t3->cfg_.ffn_dim         = (int32_t)read_u32(gguf, "chatterbox_t3.feed_forward_length",    4096);
    t3->cfg_.context_len     = (int32_t)read_u32(gguf, "chatterbox_t3.context_length",         8196);
    t3->cfg_.text_vocab      = (int32_t)read_u32(gguf, "chatterbox_t3.vocab_size",            50276);
    t3->cfg_.speech_vocab    = (int32_t)read_u32(gguf, "chatterbox_t3.speech_vocab_size",      6563);
    t3->cfg_.speaker_emb_dim = (int32_t)read_u32(gguf, "chatterbox_t3.speaker_emb_dim",         256);
    t3->cfg_.head_dim        = t3->cfg_.embed_dim / t3->cfg_.n_heads;
    t3->cfg_.layer_norm_eps  = read_f32(gguf, "chatterbox_t3.attention.layer_norm_epsilon", 1e-5f);

    auto get = [&](const char* name) -> ggml_tensor* {
        ggml_tensor* t = model->find_tensor(name);
        if (!t) std::fprintf(stderr, "T3::load: missing tensor %s\n", name);
        return t;
    };

    t3->text_emb_      = get("token_embd.text.weight");
    t3->speech_emb_    = get("token_embd.speech.weight");
    t3->pos_embd_      = get("position_embd.weight");
    t3->cond_enc_w_    = get("cond_enc.weight");
    t3->cond_enc_b_    = get("cond_enc.bias");
    t3->out_norm_w_    = get("output_norm.weight");
    t3->out_norm_b_    = get("output_norm.bias");
    t3->speech_head_w_ = get("speech_head.weight");
    t3->speech_head_b_ = get("speech_head.bias");

    if (!t3->text_emb_ || !t3->speech_emb_ || !t3->pos_embd_
        || !t3->cond_enc_w_ || !t3->cond_enc_b_
        || !t3->out_norm_w_ || !t3->out_norm_b_
        || !t3->speech_head_w_ || !t3->speech_head_b_) {
        return nullptr;
    }

    t3->layers_.resize(static_cast<size_t>(t3->cfg_.n_layers));
    for (int i = 0; i < t3->cfg_.n_layers; ++i) {
        Layer& L = t3->layers_[i];
        char nm[64];
        auto fetch = [&](const char* fmt) {
            std::snprintf(nm, sizeof(nm), fmt, i);
            return get(nm);
        };
        L.ln1_w  = fetch("blk.%d.attn_norm.weight");
        L.ln1_b  = fetch("blk.%d.attn_norm.bias");
        L.qkv_w  = fetch("blk.%d.attn_qkv.weight");
        L.qkv_b  = fetch("blk.%d.attn_qkv.bias");
        L.out_w  = fetch("blk.%d.attn_output.weight");
        L.out_b  = fetch("blk.%d.attn_output.bias");
        L.ln2_w  = fetch("blk.%d.ffn_norm.weight");
        L.ln2_b  = fetch("blk.%d.ffn_norm.bias");
        L.fc_w   = fetch("blk.%d.ffn_up.weight");
        L.fc_b   = fetch("blk.%d.ffn_up.bias");
        L.proj_w = fetch("blk.%d.ffn_down.weight");
        L.proj_b = fetch("blk.%d.ffn_down.bias");
        if (!L.ln1_w || !L.ln1_b || !L.qkv_w || !L.qkv_b
            || !L.out_w || !L.out_b || !L.ln2_w || !L.ln2_b
            || !L.fc_w || !L.fc_b || !L.proj_w || !L.proj_b) {
            return nullptr;
        }
    }

    t3->backend_ = chatterbox::default_backend();
    if (!t3->backend_) {
        std::fprintf(stderr, "T3::load: ggml_backend_cpu_init failed\n");
        return nullptr;
    }

    // One scheduler reused across all forward_impl calls.
    t3->sched_ = chatterbox::make_sched(GRAPH_MAX_NODES);
    if (!t3->sched_) {
        std::fprintf(stderr, "T3::load: make_sched failed\n");
        return nullptr;
    }

    // Pre-transpose qkv/out/fc/proj weights once (GGUF stores them as
    // (out, in); mul_mat wants (in, out)). Saves a per-step
    // cont(transpose(.)) on each of the 24 layers' 4 weights.
    {
        const int n_layers = t3->cfg_.n_layers;
        const int per      = 4;  // qkv, out, fc, proj
        ggml_init_params tp{};
        tp.mem_size = static_cast<size_t>(n_layers * per + 8)
                      * ggml_tensor_overhead();
        tp.no_alloc = true;
        t3->tw_ctx_ = ggml_init(tp);
        if (!t3->tw_ctx_) {
            std::fprintf(stderr, "T3::load: tw_ctx init failed\n");
            return nullptr;
        }
        // Create persistent transposed tensors (in, out).
        std::vector<ggml_tensor*> dst;
        dst.reserve(static_cast<size_t>(n_layers) * per);
        auto mkT = [&](ggml_tensor* w) -> ggml_tensor* {
            ggml_tensor* t = ggml_new_tensor_2d(t3->tw_ctx_, w->type,
                                                  w->ne[1], w->ne[0]);
            dst.push_back(t);
            return t;
        };
        for (int i = 0; i < n_layers; ++i) {
            const Layer& L = t3->layers_[i];
            mkT(L.qkv_w); mkT(L.out_w); mkT(L.fc_w); mkT(L.proj_w);
        }
        t3->tw_buffer_ =
            ggml_backend_alloc_ctx_tensors(t3->tw_ctx_, t3->backend_);
        if (!t3->tw_buffer_) {
            std::fprintf(stderr, "T3::load: tw_buffer alloc failed\n");
            return nullptr;
        }

        // Build a one-shot graph that transposes every weight, compute it,
        // then device-copy each result into the persistent tensor.
        const size_t gbuf =
            ggml_tensor_overhead() * (n_layers * per * 4 + 64)
            + ggml_graph_overhead_custom(n_layers * per * 4 + 64, false);
        std::vector<uint8_t> gmem(gbuf);
        ggml_init_params gp{};
        gp.mem_size   = gmem.size();
        gp.mem_buffer = gmem.data();
        gp.no_alloc   = true;
        ggml_context* gctx = ggml_init(gp);
        ggml_cgraph* tg = ggml_new_graph_custom(gctx, n_layers * per * 4 + 64, false);
        std::vector<ggml_tensor*> outs;
        outs.reserve(dst.size());
        auto add_src = [&](ggml_tensor* w) {
            ggml_tensor* t = ggml_cont(gctx, ggml_transpose(gctx, w));
            ggml_set_output(t);
            ggml_build_forward_expand(tg, t);
            outs.push_back(t);
        };
        for (int i = 0; i < n_layers; ++i) {
            const Layer& L = t3->layers_[i];
            add_src(L.qkv_w); add_src(L.out_w); add_src(L.fc_w); add_src(L.proj_w);
        }
        ggml_backend_sched_reset(t3->sched_);
        if (!ggml_backend_sched_alloc_graph(t3->sched_, tg) ||
            ggml_backend_sched_graph_compute(t3->sched_, tg)
                != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "T3::load: weight pre-transpose failed\n");
            ggml_free(gctx);
            return nullptr;
        }
        for (size_t i = 0; i < dst.size(); ++i) {
            ggml_backend_tensor_copy(outs[i], dst[i]);
        }
        ggml_backend_sched_reset(t3->sched_);
        ggml_free(gctx);

        // Repoint layer weights to the transposed tensors.
        for (int i = 0; i < n_layers; ++i) {
            Layer& L = t3->layers_[i];
            L.qkv_w  = dst[static_cast<size_t>(i) * per + 0];
            L.out_w  = dst[static_cast<size_t>(i) * per + 1];
            L.fc_w   = dst[static_cast<size_t>(i) * per + 2];
            L.proj_w = dst[static_cast<size_t>(i) * per + 3];
        }
    }

    // KV cache: a flat fp32 buffer per K/V, sized n_layers * max_context *
    // embed_dim. Layer il owns the slab [il*max_ctx*D, (il+1)*max_ctx*D).
    // Position p within a layer is the slab + p*D for D contiguous elements.
    {
        ggml_init_params p{};
        p.mem_size   = 4 * ggml_tensor_overhead() + 512;
        p.no_alloc   = true;
        t3->cache_ctx_ = ggml_init(p);
        if (!t3->cache_ctx_) {
            std::fprintf(stderr, "T3::load: KV cache ctx init failed\n");
            return nullptr;
        }
        const int64_t total =
            (int64_t)t3->cfg_.n_layers * max_context * t3->cfg_.embed_dim;
        t3->cache_k_ = ggml_new_tensor_1d(t3->cache_ctx_, GGML_TYPE_F32, total);
        t3->cache_v_ = ggml_new_tensor_1d(t3->cache_ctx_, GGML_TYPE_F32, total);
        t3->cache_buffer_ =
            ggml_backend_alloc_ctx_tensors(t3->cache_ctx_, t3->backend_);
        if (!t3->cache_buffer_) {
            std::fprintf(stderr,
                         "T3::load: KV cache buffer alloc failed (%lld bytes per buffer)\n",
                         (long long)(total * sizeof(float)));
            return nullptr;
        }
    }
    return t3;
}

void T3::reset_cache() { n_past_ = 0; }

std::vector<float> T3::prefill(
    const std::vector<float>& speaker_emb,
    const std::vector<int32_t>& cond_prompt_speech_tokens,
    const std::vector<int32_t>& text_tokens) {
    reset_cache();
    return forward_impl(/*has_cond_prefix=*/true, speaker_emb,
                        cond_prompt_speech_tokens, text_tokens, text_emb_);
}

std::vector<float> T3::append_text(const std::vector<int32_t>& tokens) {
    if (n_past_ == 0) {
        std::fprintf(stderr,
                     "T3::append_text: cache empty; call prefill first\n");
        return {};
    }
    return forward_impl(/*has_cond_prefix=*/false,
                        /*speaker_emb=*/{}, /*prompt=*/{}, tokens, text_emb_);
}

std::vector<float> T3::append_speech(const std::vector<int32_t>& tokens) {
    if (n_past_ == 0) {
        std::fprintf(stderr,
                     "T3::append_speech: cache empty; call prefill first\n");
        return {};
    }
    return forward_impl(/*has_cond_prefix=*/false,
                        /*speaker_emb=*/{}, /*prompt=*/{}, tokens, speech_emb_);
}

std::vector<int32_t> T3::generate(const GenParams& params) {
    if (n_past_ == 0) {
        std::fprintf(stderr,
                     "T3::generate: cache empty; call prefill first\n");
        return {};
    }
    if (params.max_tokens <= 0) return {};

    Sampler sampler(params.sampling);
    std::vector<int32_t> generated;
    generated.reserve(static_cast<size_t>(params.max_tokens));

    const bool prof = t3_profile_enabled();
    if (prof) g_t3_stats = T3StepStats{};
    auto t_sample_total = std::chrono::steady_clock::now();
    double sample_ms = 0.0;

    // Feed the start-of-speech token once; the returned logits cover the
    // first sampled token's position.
    std::vector<float> logits = append_speech({params.start_token});
    if (logits.empty()) return {};

    for (int i = 0; i < params.max_tokens; ++i) {
        auto t_s0 = std::chrono::steady_clock::now();
        const int32_t next = sampler.sample(logits, generated);
        if (prof) sample_ms += ms_since(t_s0);
        if (next < 0 || next >= params.vocab_limit) break;
        generated.push_back(next);
        if (n_past_ >= max_context_) break;  // ran out of cache room
        logits = append_speech({next});
        if (logits.empty()) break;
    }
    if (prof) {
        const double total = ms_since(t_sample_total);
        const int N = std::max(1, g_t3_stats.n_steps);
        std::printf("[t3] %d steps, %.2f ms total (%.2f ms/step):\n", N, total, total / N);
        std::printf("  build      %8.2f ms  (%5.2f ms/step)\n", g_t3_stats.build_ms,      g_t3_stats.build_ms / N);
        std::printf("  sched      %8.2f ms  (%5.2f ms/step)\n", g_t3_stats.sched_ms,      g_t3_stats.sched_ms / N);
        std::printf("  set_inputs %8.2f ms  (%5.2f ms/step)\n", g_t3_stats.set_inputs_ms, g_t3_stats.set_inputs_ms / N);
        std::printf("  compute    %8.2f ms  (%5.2f ms/step)\n", g_t3_stats.compute_ms,    g_t3_stats.compute_ms / N);
        std::printf("  readback   %8.2f ms  (%5.2f ms/step)\n", g_t3_stats.readback_ms,   g_t3_stats.readback_ms / N);
        std::printf("  sample     %8.2f ms  (%5.2f ms/step)\n", sample_ms,                sample_ms / N);
    }
    return generated;
}

std::vector<float> T3::forward_impl(
    bool has_cond_prefix,
    const std::vector<float>& speaker_emb,
    const std::vector<int32_t>& cond_prompt_speech_tokens,
    const std::vector<int32_t>& text_tokens,
    ggml_tensor* emb_table) {
    const int L_text   = static_cast<int>(text_tokens.size());
    const int L_prompt = has_cond_prefix
        ? static_cast<int>(cond_prompt_speech_tokens.size()) : 0;
    const int N_new = L_text + L_prompt + (has_cond_prefix ? 1 : 0);
    if (N_new <= 0) {
        std::fprintf(stderr, "T3::forward_impl: no tokens to process\n");
        return {};
    }
    if (has_cond_prefix &&
        static_cast<int>(speaker_emb.size()) != cfg_.speaker_emb_dim) {
        std::fprintf(stderr,
                     "T3::forward_impl: speaker_emb size %zu != %d (prefill)\n",
                     speaker_emb.size(), cfg_.speaker_emb_dim);
        return {};
    }
    const int n_past  = n_past_;
    const int N_total = n_past + N_new;
    if (N_total > max_context_) {
        std::fprintf(stderr,
                     "T3::forward_impl: N_total=%d exceeds max_context=%d\n",
                     N_total, max_context_);
        return {};
    }

    const int D  = cfg_.embed_dim;
    const int H  = cfg_.n_heads;
    const int Hd = cfg_.head_dim;
    const float eps = cfg_.layer_norm_eps;

    // Build graph in a no-alloc context. Compute tensor data is then
    // allocated by the per-call gallocr; KV cache lives in its own
    // pre-allocated buffer.
    const bool prof = t3_profile_enabled();
    auto t_build0 = std::chrono::steady_clock::now();

    const size_t buf_size =
        ggml_tensor_overhead() * GRAPH_MAX_NODES
        + ggml_graph_overhead_custom(GRAPH_MAX_NODES, false);
    std::vector<uint8_t> buf(buf_size);

    ggml_init_params params{};
    params.mem_size   = buf.size();
    params.mem_buffer = buf.data();
    params.no_alloc   = true;

    ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        std::fprintf(stderr, "T3::forward_impl: ggml_init failed\n");
        return {};
    }
    ggml_cgraph* gf = ggml_new_graph_custom(ctx, GRAPH_MAX_NODES, false);

    // --- Inputs ---
    ggml_tensor* tokens = nullptr;
    if (L_text > 0) {
        tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, L_text);
        ggml_set_name(tokens, "tokens");
        ggml_set_input(tokens);
    }

    ggml_tensor* spk_in = nullptr;
    ggml_tensor* prompt_in = nullptr;
    if (has_cond_prefix) {
        spk_in = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, cfg_.speaker_emb_dim);
        ggml_set_name(spk_in, "speaker_emb");
        ggml_set_input(spk_in);

        if (L_prompt > 0) {
            prompt_in = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, L_prompt);
            ggml_set_name(prompt_in, "cond_prompt_speech");
            ggml_set_input(prompt_in);
        }
    }

    ggml_tensor* positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, N_new);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);

    // --- Build the new-token embedding slice ---
    // Order: [cond_spkr(1), cond_prompt_speech(L_prompt), tokens(L_text)]
    // when has_cond_prefix; otherwise just [tokens].
    ggml_tensor* seq = nullptr;

    if (has_cond_prefix) {
        // cond_spkr = cond_enc(spk_in) -> (D, 1)
        ggml_tensor* cond_emb = ggml_mul_mat(ctx, cond_enc_w_, spk_in);
        cond_emb = ggml_add(ctx, cond_emb, cond_enc_b_);
        seq = ggml_reshape_2d(ctx, cond_emb, D, 1);

        // cond_prompt_speech_emb = speech_emb[prompt_in] -> (D, L_prompt)
        if (L_prompt > 0) {
            ggml_tensor* prompt_h = ggml_get_rows(ctx, speech_emb_, prompt_in);
            prompt_h = ggml_cast(ctx, prompt_h, GGML_TYPE_F32);
            seq = ggml_concat(ctx, seq, prompt_h, 1);
        }
    }
    if (L_text > 0) {
        ggml_tensor* tok_h = ggml_get_rows(ctx, emb_table, tokens);
        tok_h = ggml_cast(ctx, tok_h, GGML_TYPE_F32);
        seq = seq ? ggml_concat(ctx, seq, tok_h, 1) : tok_h;
    }
    // seq: (D, N_new)

    // Add positional embedding (absolute positions [n_past, n_past + N_new)).
    ggml_tensor* pos = ggml_get_rows(ctx, pos_embd_, positions);
    pos = ggml_cast(ctx, pos, GGML_TYPE_F32);
    ggml_tensor* h = ggml_add(ctx, seq, pos);

    // --- Transformer blocks ---
    const size_t k_elt = ggml_element_size(cache_k_);  // 4 bytes (fp32)

    for (int il = 0; il < cfg_.n_layers; ++il) {
        const Layer& L = layers_[il];
        ggml_tensor* cur;

        // Pre-attn LN
        cur = ggml_norm(ctx, h, eps);
        cur = ggml_add(ctx, ggml_mul(ctx, cur, L.ln1_w), L.ln1_b);

        // QKV proj. Weight pre-transposed to (in, out) at load.
        cur = ggml_mul_mat(ctx, L.qkv_w, cur);
        cur = ggml_add(ctx, cur, L.qkv_b);
        // cur: (3D, N_new)

        ggml_tensor* Qcur = ggml_view_2d(ctx, cur, D, N_new, cur->nb[1], 0 * D * cur->nb[0]);
        ggml_tensor* Kcur = ggml_view_2d(ctx, cur, D, N_new, cur->nb[1], 1 * D * cur->nb[0]);
        ggml_tensor* Vcur = ggml_view_2d(ctx, cur, D, N_new, cur->nb[1], 2 * D * cur->nb[0]);

        // ---- KV cache writes ----
        // Each layer owns max_context_ * D contiguous elements; positions
        // n_past..n_past+N_new go to that slab.
        const size_t layer_offset_bytes =
            (size_t)il * max_context_ * D * k_elt;
        const size_t pos_offset_bytes = (size_t)n_past * D * k_elt;
        const size_t write_offset = layer_offset_bytes + pos_offset_bytes;
        ggml_tensor* k_write = ggml_view_2d(ctx, cache_k_,
            D, N_new, D * k_elt, write_offset);
        ggml_tensor* v_write = ggml_view_2d(ctx, cache_v_,
            D, N_new, D * k_elt, write_offset);
        ggml_build_forward_expand(gf, ggml_cpy(ctx, Kcur, k_write));
        ggml_build_forward_expand(gf, ggml_cpy(ctx, Vcur, v_write));

        // ---- Q, K, V for attention ----
        // Q has only the new tokens (N_new). K and V cover the full
        // sequence so far (N_total).
        ggml_tensor* Q = ggml_permute(ctx,
            ggml_cont_3d(ctx, Qcur, Hd, H, N_new), 0, 2, 1, 3);

        ggml_tensor* k_view = ggml_view_1d(ctx, cache_k_,
            N_total * D, layer_offset_bytes);
        ggml_tensor* K = ggml_permute(ctx,
            ggml_reshape_3d(ctx, k_view, Hd, H, N_total),
            0, 2, 1, 3);

        ggml_tensor* v_view = ggml_view_1d(ctx, cache_v_,
            N_total * D, layer_offset_bytes);
        ggml_tensor* V_trans = ggml_cont_3d(ctx,
            ggml_permute(ctx,
                ggml_reshape_3d(ctx, v_view, Hd, H, N_total),
                1, 2, 0, 3),
            N_total, Hd, H);

        // Scaled, masked, softmaxed scores.
        ggml_tensor* KQ = ggml_mul_mat(ctx, K, Q);  // (N_total, N_new, H)
        KQ = ggml_scale(ctx, KQ, 1.0f / std::sqrt(static_cast<float>(Hd)));
        // ggml_diag_mask_inf with n_past shifts the mask: column j is masked
        // when j > i + n_past where i is the row index. Exactly what we want
        // for incremental attention.
        KQ = ggml_diag_mask_inf(ctx, KQ, n_past);
        KQ = ggml_soft_max(ctx, KQ);

        ggml_tensor* KQV = ggml_mul_mat(ctx, V_trans, KQ);              // (Hd, N_new, H)
        ggml_tensor* KQV_merged = ggml_permute(ctx, KQV, 0, 2, 1, 3);   // (Hd, H, N_new)
        cur = ggml_cont_2d(ctx, KQV_merged, D, N_new);                  // (D, N_new)

        // Attn output proj (weight pre-transposed at load).
        cur = ggml_mul_mat(ctx, L.out_w, cur);
        cur = ggml_add(ctx, cur, L.out_b);

        ggml_tensor* attn_residual = ggml_add(ctx, cur, h);

        // Pre-FFN LN
        cur = ggml_norm(ctx, attn_residual, eps);
        cur = ggml_add(ctx, ggml_mul(ctx, cur, L.ln2_w), L.ln2_b);

        // FFN (weights pre-transposed at load).
        cur = ggml_mul_mat(ctx, L.fc_w, cur);
        cur = ggml_add(ctx, cur, L.fc_b);
        cur = ggml_gelu(ctx, cur);
        cur = ggml_mul_mat(ctx, L.proj_w, cur);
        cur = ggml_add(ctx, cur, L.proj_b);

        h = ggml_add(ctx, cur, attn_residual);
    }

    // Final LN
    h = ggml_norm(ctx, h, eps);
    h = ggml_add(ctx, ggml_mul(ctx, h, out_norm_w_), out_norm_b_);

    // Last position (col N_new - 1 within the new chunk).
    ggml_tensor* last = ggml_view_1d(ctx, h, D, (N_new - 1) * h->nb[1]);
    ggml_tensor* logits = ggml_mul_mat(ctx, speech_head_w_, last);
    logits = ggml_add(ctx, logits, speech_head_b_);

    ggml_set_name(logits, "logits");
    ggml_set_output(logits);
    ggml_build_forward_expand(gf, logits);

    if (prof) g_t3_stats.build_ms += ms_since(t_build0);
    auto t_sched0 = std::chrono::steady_clock::now();
    // Reuse the persistent scheduler (created in load()). Reset clears
    // the previous graph's allocations before we allocate this one.
    //
    // Note: pinning `logits` to CPU does NOT propagate backward
    // through the transformer stack — the KV cache tensors live in
    // a fixed-backend buffer that sched can't reassign. The
    // transformer compute stays on Vulkan; only the final
    // speech_head projection would move to CPU. That's not enough
    // to recover the fp32 parity reference (drift accumulates across
    // 24 blocks).
    //
    // For T3 we accept the fp16 drift on Vulkan: top-5 token ids
    // stay identical to the CPU/fp32 reference, autoregressive
    // sampling produces correct sequences, and tolerances in the
    // forward parity test are widened to match. See t3_forward_test.
    ggml_backend_sched_t sched = sched_;
    ggml_backend_sched_reset(sched);
    if (!sched || !ggml_backend_sched_alloc_graph(sched, gf)) {
        std::fprintf(stderr, "T3::forward_impl: gallocr_alloc_graph failed\n");
        ggml_free(ctx);
        return {};
    }

    if (prof) g_t3_stats.sched_ms += ms_since(t_sched0);
    auto t_set0 = std::chrono::steady_clock::now();
    if (tokens) {
        ggml_backend_tensor_set(tokens, text_tokens.data(), 0,
                                L_text * sizeof(int32_t));
    }
    if (spk_in) {
        ggml_backend_tensor_set(spk_in, speaker_emb.data(), 0,
                                cfg_.speaker_emb_dim * sizeof(float));
    }
    if (prompt_in) {
        ggml_backend_tensor_set(prompt_in,
                                cond_prompt_speech_tokens.data(), 0,
                                L_prompt * sizeof(int32_t));
    }
    std::vector<int32_t> pos_ids(N_new);
    for (int i = 0; i < N_new; ++i) pos_ids[i] = n_past + i;
    ggml_backend_tensor_set(positions, pos_ids.data(), 0,
                            N_new * sizeof(int32_t));

    if (prof) g_t3_stats.set_inputs_ms += ms_since(t_set0);
    auto t_compute0 = std::chrono::steady_clock::now();
    if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "T3::forward_impl: backend_graph_compute failed\n");
        ggml_free(ctx);
        return {};
    }
    if (prof) g_t3_stats.compute_ms += ms_since(t_compute0);
    auto t_read0 = std::chrono::steady_clock::now();

    std::vector<float> result(static_cast<size_t>(cfg_.speech_vocab));
    ggml_backend_tensor_get(logits, result.data(), 0,
                            result.size() * sizeof(float));

    ggml_free(ctx);

    if (prof) {
        g_t3_stats.readback_ms += ms_since(t_read0);
        g_t3_stats.n_steps += 1;
    }
    n_past_ += N_new;
    return result;
}

}  // namespace chatterbox
