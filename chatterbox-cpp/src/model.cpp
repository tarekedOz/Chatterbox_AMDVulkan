#include "model.h"

#include "backend.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <cstdio>
#include <cstring>
#include <vector>

namespace chatterbox {

Model::~Model() {
    // Order: free the backend buffer (releases tensor data) BEFORE
    // freeing the ggml context (which owns the tensor structs).
    if (buf_)  ggml_backend_buffer_free(buf_);
    if (ctx_)  ggml_free(ctx_);
    if (gguf_) gguf_free(gguf_);
}

ggml_tensor* Model::find_tensor(const std::string& n) const {
    for (const auto& t : tensors) {
        if (t.name == n) return t.tensor;
    }
    return nullptr;
}

namespace {

std::string read_string_key(gguf_context* gguf, const char* key) {
    const int64_t idx = gguf_find_key(gguf, key);
    if (idx < 0) return {};
    return gguf_get_val_str(gguf, idx);
}

}  // namespace

std::unique_ptr<Model> load_model(const std::string& path) {
    auto m = std::unique_ptr<Model>(new Model());

    // 1. Read GGUF metadata only — `no_alloc=true` skips allocating
    //    CPU memory for tensor data; we'll allocate via the backend
    //    below.
    ggml_context* meta_ctx = nullptr;
    gguf_init_params params{};
    params.no_alloc = true;
    params.ctx      = &meta_ctx;

    m->gguf_ = gguf_init_from_file(path.c_str(), params);
    if (!m->gguf_) {
        std::fprintf(stderr, "chatterbox: gguf_init_from_file failed for %s\n",
                     path.c_str());
        return nullptr;
    }
    m->ctx_ = meta_ctx;

    m->arch        = read_string_key(m->gguf_, "general.architecture");
    m->name        = read_string_key(m->gguf_, "general.name");
    m->description = read_string_key(m->gguf_, "general.description");

    if (m->arch.empty()) {
        std::fprintf(stderr,
                     "chatterbox: %s has no general.architecture key\n",
                     path.c_str());
        return nullptr;
    }

    // 2. Allocate all tensor data in the process-wide default backend.
    //    With Vulkan available, this puts the weights on the GPU
    //    (or in UMA system memory mapped to the GPU on Strix Halo).
    ggml_backend_t backend = default_backend();
    m->buf_ = ggml_backend_alloc_ctx_tensors(m->ctx_, backend);
    if (!m->buf_) {
        std::fprintf(stderr,
                     "chatterbox: ggml_backend_alloc_ctx_tensors failed "
                     "for %s\n", path.c_str());
        return nullptr;
    }

    // 3. Read tensor bytes from the file and upload to the backend
    //    via ggml_backend_tensor_set. Re-open the file rather than
    //    threading mmap state through gguf — keeps the loader simple
    //    and works the same on all platforms.
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        std::fprintf(stderr, "chatterbox: cannot reopen %s for tensor data\n",
                     path.c_str());
        return nullptr;
    }
    const size_t data_offset = gguf_get_data_offset(m->gguf_);
    const int64_t n = gguf_get_n_tensors(m->gguf_);
    m->n_tensors = static_cast<size_t>(n);
    m->tensors.reserve(m->n_tensors);

    // Scratch buffer for tensor bytes. Sized to the largest tensor
    // we encounter; resized as needed.
    std::vector<uint8_t> scratch;

    for (int64_t i = 0; i < n; ++i) {
        const char* tname = gguf_get_tensor_name(m->gguf_, i);
        ggml_tensor* t = ggml_get_tensor(m->ctx_, tname);
        if (!t) {
            std::fprintf(stderr,
                         "chatterbox: tensor index %lld (%s) missing in ctx\n",
                         static_cast<long long>(i), tname ? tname : "?");
            std::fclose(f);
            return nullptr;
        }
        const size_t tbytes = ggml_nbytes(t);
        const size_t toff   = data_offset + gguf_get_tensor_offset(m->gguf_, i);

        if (scratch.size() < tbytes) scratch.resize(tbytes);
        if (std::fseek(f, static_cast<long>(toff), SEEK_SET) != 0) {
            std::fprintf(stderr,
                         "chatterbox: fseek to %zu for tensor %s failed\n",
                         toff, tname);
            std::fclose(f);
            return nullptr;
        }
        if (std::fread(scratch.data(), 1, tbytes, f) != tbytes) {
            std::fprintf(stderr,
                         "chatterbox: short read for tensor %s (%zu bytes)\n",
                         tname, tbytes);
            std::fclose(f);
            return nullptr;
        }
        ggml_backend_tensor_set(t, scratch.data(), 0, tbytes);

        TensorInfo info;
        info.name   = tname;
        info.tensor = t;
        for (int d = 0; d < 4; ++d) info.ne[d] = t->ne[d];
        info.dtype  = static_cast<int>(t->type);
        info.nbytes = tbytes;
        m->total_tensor_bytes += tbytes;
        m->tensors.push_back(std::move(info));
    }
    std::fclose(f);

    return m;
}

}  // namespace chatterbox
