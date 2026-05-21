#include "voice_pack.h"

#include "wav_io.h"

#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace chatterbox {

namespace {

constexpr const char* ARCH         = "chatterbox_voices";
constexpr const char* KEY_ARCH     = "general.architecture";
constexpr const char* KEY_NAME     = "general.name";
constexpr const char* KEY_COUNT    = "voices.count";
constexpr const char* KEY_NAMES    = "voices.names";

std::string voice_key(int idx, const char* field) {
    char buf[128];
    std::snprintf(buf, sizeof(buf), "voice.%d.%s", idx, field);
    return std::string(buf);
}

}  // namespace


// ----------------------------------------------------------------------------
// Loader
// ----------------------------------------------------------------------------

std::unique_ptr<VoicePack> VoicePack::load(const std::string& path) {
    ggml_context* ggml_ctx = nullptr;
    gguf_init_params params{};
    params.no_alloc = false;
    params.ctx      = &ggml_ctx;
    gguf_context* gguf = gguf_init_from_file(path.c_str(), params);
    if (!gguf) {
        std::fprintf(stderr, "VoicePack::load: cannot open %s\n", path.c_str());
        return nullptr;
    }

    // Validate arch.
    {
        const int64_t id = gguf_find_key(gguf, KEY_ARCH);
        if (id < 0
            || std::strcmp(gguf_get_val_str(gguf, id), ARCH) != 0) {
            std::fprintf(stderr, "VoicePack::load: bad arch (expected %s)\n", ARCH);
            gguf_free(gguf);
            if (ggml_ctx) ggml_free(ggml_ctx);
            return nullptr;
        }
    }
    const int64_t count_id = gguf_find_key(gguf, KEY_COUNT);
    if (count_id < 0) {
        std::fprintf(stderr, "VoicePack::load: missing %s\n", KEY_COUNT);
        gguf_free(gguf);
        if (ggml_ctx) ggml_free(ggml_ctx);
        return nullptr;
    }
    const uint32_t N = gguf_get_val_u32(gguf, count_id);

    auto vp = std::unique_ptr<VoicePack>(new VoicePack());
    vp->names_.resize(N);
    vp->conds_.resize(N);

    // Voice names.
    const int64_t names_id = gguf_find_key(gguf, KEY_NAMES);
    if (names_id < 0) {
        std::fprintf(stderr, "VoicePack::load: missing %s\n", KEY_NAMES);
        gguf_free(gguf);
        if (ggml_ctx) ggml_free(ggml_ctx);
        return nullptr;
    }
    if (gguf_get_arr_n(gguf, names_id) != N) {
        std::fprintf(stderr,
                     "VoicePack::load: voices.names size %zu != count %u\n",
                     gguf_get_arr_n(gguf, names_id), N);
        gguf_free(gguf);
        if (ggml_ctx) ggml_free(ggml_ctx);
        return nullptr;
    }
    for (uint32_t i = 0; i < N; ++i) {
        vp->names_[i] = gguf_get_arr_str(gguf, names_id, i);
    }

    // Tensors.
    for (uint32_t i = 0; i < N; ++i) {
        auto& c = vp->conds_[i];
        const auto k_spk_256 = voice_key(i, "spk_emb_256");
        const auto k_spk_192 = voice_key(i, "spk_emb_192");
        const auto k_tokens  = voice_key(i, "prompt_tokens");
        const auto k_feat    = voice_key(i, "prompt_feat");
        ggml_tensor* t_spk_256 = ggml_get_tensor(ggml_ctx, k_spk_256.c_str());
        ggml_tensor* t_spk_192 = ggml_get_tensor(ggml_ctx, k_spk_192.c_str());
        ggml_tensor* t_tokens  = ggml_get_tensor(ggml_ctx, k_tokens.c_str());
        ggml_tensor* t_feat    = ggml_get_tensor(ggml_ctx, k_feat.c_str());
        if (!t_spk_256 || !t_spk_192 || !t_tokens || !t_feat) {
            std::fprintf(stderr, "VoicePack::load: voice %u missing tensor(s)\n", i);
            gguf_free(gguf);
            if (ggml_ctx) ggml_free(ggml_ctx);
            return nullptr;
        }
        const size_t n_spk_256 = ggml_nelements(t_spk_256);
        const size_t n_spk_192 = ggml_nelements(t_spk_192);
        const size_t n_tokens  = ggml_nelements(t_tokens);
        const size_t n_feat    = ggml_nelements(t_feat);
        // VoicePack uses its own ggml_context with `no_alloc=false`,
        // so the tensor data lives in plain host memory — direct
        // `->data` access is safe here (no backend involved).
        c.spk_emb_256.assign(static_cast<float*>(t_spk_256->data),
                              static_cast<float*>(t_spk_256->data) + n_spk_256);
        c.spk_emb_192.assign(static_cast<float*>(t_spk_192->data),
                              static_cast<float*>(t_spk_192->data) + n_spk_192);
        c.prompt_tokens.assign(static_cast<int32_t*>(t_tokens->data),
                                static_cast<int32_t*>(t_tokens->data) + n_tokens);
        c.prompt_feat.assign(static_cast<float*>(t_feat->data),
                              static_cast<float*>(t_feat->data) + n_feat);
        // prompt_feat shape is (T_mel, 80) — recover T_mel from total.
        c.prompt_feat_T = static_cast<int>(n_feat / 80);
    }
    gguf_free(gguf);
    if (ggml_ctx) ggml_free(ggml_ctx);
    return vp;
}

bool VoicePack::get_conditioning(const std::string& name,
                                    Conditioning& out) const {
    auto it = std::find(names_.begin(), names_.end(), name);
    if (it == names_.end()) return false;
    const size_t idx = static_cast<size_t>(it - names_.begin());
    out = conds_[idx];
    return true;
}


// ----------------------------------------------------------------------------
// Packer
// ----------------------------------------------------------------------------

int VoicePack::pack(Chatterbox& cb,
                       const std::string& voice_dir,
                       const std::string& out_path,
                       bool verbose) {
    if (!fs::is_directory(voice_dir)) {
        std::fprintf(stderr, "pack: %s is not a directory\n", voice_dir.c_str());
        return -1;
    }

    // Collect *.wav files sorted by name.
    std::vector<fs::path> wav_paths;
    for (const auto& entry : fs::directory_iterator(voice_dir)) {
        if (!entry.is_regular_file()) continue;
        const auto ext = entry.path().extension().string();
        if (ext != ".wav" && ext != ".WAV") continue;
        wav_paths.push_back(entry.path());
    }
    std::sort(wav_paths.begin(), wav_paths.end());
    if (wav_paths.empty()) {
        std::fprintf(stderr, "pack: no .wav files in %s\n", voice_dir.c_str());
        return -1;
    }

    std::vector<std::string>   names;
    std::vector<Conditioning>  conds;
    names.reserve(wav_paths.size());
    conds.reserve(wav_paths.size());

    for (size_t i = 0; i < wav_paths.size(); ++i) {
        const auto& p = wav_paths[i];
        const std::string name = p.stem().string();
        std::vector<float> wav;
        int sr = 0;
        if (!read_wav_mono(p.string(), wav, sr)) {
            std::fprintf(stderr, "pack: read_wav_mono failed for %s\n",
                          p.string().c_str());
            continue;
        }
        if (verbose) {
            std::printf("[%zu/%zu] %s (%zu samp @ %d Hz, %.2f s)\n",
                         i + 1, wav_paths.size(), name.c_str(),
                         wav.size(), sr,
                         static_cast<double>(wav.size()) / sr);
        }
        Conditioning c;
        if (!cb.extract_conditioning(wav, sr, c)) {
            std::fprintf(stderr, "pack: extract_conditioning failed for %s\n",
                          name.c_str());
            continue;
        }
        names.push_back(name);
        conds.push_back(std::move(c));
    }
    if (names.empty()) {
        std::fprintf(stderr, "pack: no voices extracted\n");
        return -1;
    }

    // Build a ggml context just large enough to hold the tensor
    // metadata (no_alloc=true means we manage data buffers ourselves).
    const size_t n_tensors_per_voice = 4;
    const size_t n_tensors           = n_tensors_per_voice * names.size();
    const size_t ctx_size            = (n_tensors + 16) * ggml_tensor_overhead();
    ggml_init_params gparams{};
    gparams.mem_size   = ctx_size;
    gparams.mem_buffer = nullptr;
    gparams.no_alloc   = true;
    ggml_context* gctx = ggml_init(gparams);
    if (!gctx) {
        std::fprintf(stderr, "pack: ggml_init failed\n");
        return -1;
    }

    // Build GGUF context.
    gguf_context* gguf = gguf_init_empty();
    gguf_set_val_str(gguf, KEY_ARCH, ARCH);
    gguf_set_val_str(gguf, KEY_NAME, "chatterbox_voices_v1");
    gguf_set_val_u32(gguf, KEY_COUNT, static_cast<uint32_t>(names.size()));

    // voices.names = string array
    std::vector<const char*> name_ptrs(names.size());
    for (size_t i = 0; i < names.size(); ++i) name_ptrs[i] = names[i].c_str();
    gguf_set_arr_str(gguf, KEY_NAMES, name_ptrs.data(), name_ptrs.size());

    // Add tensors (metadata only). Data is set via gguf_set_tensor_data
    // after gguf_add_tensor.
    for (size_t i = 0; i < names.size(); ++i) {
        const auto& c = conds[i];
        // spk_emb_256
        ggml_tensor* t1 = ggml_new_tensor_1d(gctx, GGML_TYPE_F32,
                                                static_cast<int64_t>(c.spk_emb_256.size()));
        ggml_set_name(t1, voice_key(i, "spk_emb_256").c_str());
        gguf_add_tensor(gguf, t1);
        gguf_set_tensor_data(gguf, t1->name, c.spk_emb_256.data());

        // spk_emb_192
        ggml_tensor* t2 = ggml_new_tensor_1d(gctx, GGML_TYPE_F32,
                                                static_cast<int64_t>(c.spk_emb_192.size()));
        ggml_set_name(t2, voice_key(i, "spk_emb_192").c_str());
        gguf_add_tensor(gguf, t2);
        gguf_set_tensor_data(gguf, t2->name, c.spk_emb_192.data());

        // prompt_tokens
        ggml_tensor* t3 = ggml_new_tensor_1d(gctx, GGML_TYPE_I32,
                                                static_cast<int64_t>(c.prompt_tokens.size()));
        ggml_set_name(t3, voice_key(i, "prompt_tokens").c_str());
        gguf_add_tensor(gguf, t3);
        gguf_set_tensor_data(gguf, t3->name, c.prompt_tokens.data());

        // prompt_feat (T_mel, 80) ggml ne=(80, T_mel). Numpy row-major
        // bytes match ggml ne=(80, T_mel) directly.
        ggml_tensor* t4 = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, 80,
                                                static_cast<int64_t>(c.prompt_feat_T));
        ggml_set_name(t4, voice_key(i, "prompt_feat").c_str());
        gguf_add_tensor(gguf, t4);
        gguf_set_tensor_data(gguf, t4->name, c.prompt_feat.data());
    }

    if (!gguf_write_to_file(gguf, out_path.c_str(), /*only_meta=*/false)) {
        std::fprintf(stderr, "pack: gguf_write_to_file failed\n");
        gguf_free(gguf);
        ggml_free(gctx);
        return -1;
    }
    if (verbose) {
        std::printf("\nWrote %zu voices to %s\n",
                     names.size(), out_path.c_str());
    }
    gguf_free(gguf);
    ggml_free(gctx);
    return static_cast<int>(names.size());
}

}  // namespace chatterbox
