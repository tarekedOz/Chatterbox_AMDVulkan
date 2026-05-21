// Smoke test for the Phase 1.E GGUF loader.
//
// Opens a GGUF file produced by scripts/convert_chatterbox_to_gguf.py
// (or its VE / S3Gen siblings), and prints a one-screen summary:
//
//   arch, name, total tensor count, total weight bytes,
//   per-dtype tensor count, first few tensor names + shapes.
//
// Exits 0 on success, 1 on failure to load, 2 on usage error.

#include "model.h"

#include "ggml.h"

#include <cstdio>
#include <map>
#include <string>

namespace {

const char* type_name(int t) {
    switch (static_cast<ggml_type>(t)) {
        case GGML_TYPE_F32:  return "F32";
        case GGML_TYPE_F16:  return "F16";
        case GGML_TYPE_BF16: return "BF16";
        case GGML_TYPE_I8:   return "I8";
        case GGML_TYPE_I16:  return "I16";
        case GGML_TYPE_I32:  return "I32";
        case GGML_TYPE_I64:  return "I64";
        case GGML_TYPE_Q4_0: return "Q4_0";
        case GGML_TYPE_Q4_1: return "Q4_1";
        case GGML_TYPE_Q5_0: return "Q5_0";
        case GGML_TYPE_Q5_1: return "Q5_1";
        case GGML_TYPE_Q8_0: return "Q8_0";
        case GGML_TYPE_Q4_K: return "Q4_K";
        case GGML_TYPE_Q5_K: return "Q5_K";
        case GGML_TYPE_Q6_K: return "Q6_K";
        case GGML_TYPE_Q8_K: return "Q8_K";
        default:             return "?";
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "Usage: %s <path-to-gguf>\n", argv[0]);
        return 2;
    }

    auto m = chatterbox::load_model(argv[1]);
    if (!m) {
        std::fprintf(stderr, "Failed to load %s\n", argv[1]);
        return 1;
    }

    std::printf("File:        %s\n", argv[1]);
    std::printf("Arch:        %s\n", m->arch.c_str());
    if (!m->name.empty())        std::printf("Name:        %s\n", m->name.c_str());
    if (!m->description.empty()) std::printf("Description: %s\n",
                                              m->description.c_str());
    std::printf("Tensors:     %zu\n", m->n_tensors);
    std::printf("Weights:     %.2f MiB (%zu bytes)\n",
                m->total_tensor_bytes / (1024.0 * 1024.0),
                m->total_tensor_bytes);

    // Per-dtype histogram
    std::map<int, size_t> by_dtype;
    for (const auto& t : m->tensors) by_dtype[t.dtype]++;
    std::printf("By dtype:\n");
    for (const auto& [dt, count] : by_dtype) {
        std::printf("  %-6s  %zu\n", type_name(dt), count);
    }

    // First few tensor names + shapes (sanity).
    const size_t k = m->tensors.size() < 5 ? m->tensors.size() : 5;
    std::printf("First %zu tensors:\n", k);
    for (size_t i = 0; i < k; ++i) {
        const auto& t = m->tensors[i];
        std::printf("  [%-6s] [%lld, %lld, %lld, %lld]  %s\n",
                    type_name(t.dtype),
                    static_cast<long long>(t.ne[0]),
                    static_cast<long long>(t.ne[1]),
                    static_cast<long long>(t.ne[2]),
                    static_cast<long long>(t.ne[3]),
                    t.name.c_str());
    }

    return 0;
}
