#pragma once

// Minimal GGUF model loader. Phase 1.E scaffolding only — does enough to
// open one of our produced GGUFs (chatterbox_t3 / _ve / _s3gen), capture
// the arch + tensor inventory, and let a smoke test print the result.
//
// Forward passes, KV cache, sampling, etc. all come later. The structure
// here is intentionally small so it's easy to grow without rework.

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct ggml_context;
struct gguf_context;
struct ggml_tensor;
struct ggml_backend_buffer;
typedef ggml_backend_buffer* ggml_backend_buffer_t;

namespace chatterbox {

struct TensorInfo {
    std::string name;
    ggml_tensor* tensor;        // owned by the Model's ggml_context
    int64_t      ne[4];         // ggml dims (innermost first)
    int          dtype;         // ggml_type enum value
    size_t       nbytes;
};

struct Model {
    std::string arch;           // e.g. "chatterbox_t3"
    std::string name;           // GGUF general.name, optional
    std::string description;    // GGUF general.description, optional

    size_t n_tensors = 0;
    size_t total_tensor_bytes = 0;

    std::vector<TensorInfo> tensors;

    // Lifetime: load_model returns a Model whose underlying ggml + gguf
    // contexts are owned by it. Destructor frees both.
    Model() = default;
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;
    Model(Model&&) = default;
    Model& operator=(Model&&) = default;
    ~Model();

    // Direct lookup. Returns nullptr if absent. O(n).
    ggml_tensor* find_tensor(const std::string& name) const;

    // Underlying GGUF context — used by Tokenizer to read tokenizer.ggml.*
    // metadata keys without re-opening the file.
    gguf_context* gguf() const { return gguf_; }

private:
    friend std::unique_ptr<Model> load_model(const std::string&);
    gguf_context*         gguf_ = nullptr;
    ggml_context*         ctx_  = nullptr;
    // Backend buffer holding ALL tensor data. Allocated by load_model
    // via ggml_backend_alloc_ctx_tensors so tensors live in the
    // process-wide default backend's memory (Vulkan if available,
    // else CPU). Freed in ~Model.
    ggml_backend_buffer_t buf_ = nullptr;
};

// Load a GGUF file. Returns nullptr on failure (diagnostic printed to
// stderr). On success the Model owns all underlying ggml resources.
std::unique_ptr<Model> load_model(const std::string& path);

}  // namespace chatterbox
