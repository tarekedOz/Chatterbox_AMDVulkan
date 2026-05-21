#include "backend.h"

#include "ggml.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifdef GGML_USE_VULKAN
#include "ggml-vulkan.h"
#endif

namespace chatterbox {

namespace {

bool env_disables_vulkan() {
    const char* v = std::getenv("CHATTERBOX_DISABLE_VULKAN");
    return v && v[0] && v[0] != '0';
}

// One-shot init. C++ guarantees this initializer runs at most once
// across threads.
ggml_backend_t make_singleton() {
#ifdef GGML_USE_VULKAN
    if (!env_disables_vulkan()) {
        ggml_backend_t vk = ggml_backend_vk_init(0);
        if (vk) {
            std::printf("chatterbox: using Vulkan backend (%s)\n",
                        ggml_backend_name(vk));
            return vk;
        }
        std::fprintf(stderr,
                     "chatterbox: ggml_backend_vk_init failed; "
                     "falling back to CPU\n");
    }
#endif
    ggml_backend_t cpu = ggml_backend_cpu_init();
    std::printf("chatterbox: using CPU backend\n");
    return cpu;
}

}  // namespace

ggml_backend_t default_backend() {
    static ggml_backend_t s_backend = make_singleton();
    return s_backend;
}

ggml_backend_buffer_type_t default_buft() {
    return ggml_backend_get_default_buffer_type(default_backend());
}

bool is_vulkan_backend(ggml_backend_t backend) {
#ifdef GGML_USE_VULKAN
    if (!backend) return false;
    const char* nm = ggml_backend_name(backend);
    return std::strstr(nm, "Vulkan") != nullptr;
#else
    (void) backend;
    return false;
#endif
}

bool is_vulkan_active() {
    return is_vulkan_backend(default_backend());
}

ggml_backend_t cpu_backend() {
    static ggml_backend_t s_cpu = ggml_backend_cpu_init();
    return s_cpu;
}

std::vector<float> read_tensor_f32(const ggml_tensor* t) {
    if (!t) return {};
    const size_t n = ggml_nelements(t);
    if (t->type == GGML_TYPE_F32) {
        std::vector<float> out(n);
        // ggml_backend_tensor_get works whether the tensor lives in CPU
        // memory or in a device backend's buffer; the runtime picks the
        // right copy path.
        ggml_backend_tensor_get(t, out.data(), 0, n * sizeof(float));
        return out;
    }
    if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> raw(n);
        ggml_backend_tensor_get(t, raw.data(), 0,
                                  n * sizeof(ggml_fp16_t));
        std::vector<float> out(n);
        for (size_t i = 0; i < n; ++i) out[i] = ggml_fp16_to_fp32(raw[i]);
        return out;
    }
    std::fprintf(stderr,
                 "read_tensor_f32: unsupported dtype %d for tensor '%s'\n",
                 static_cast<int>(t->type),
                 t->name[0] ? t->name : "?");
    return {};
}

ggml_backend_sched_t make_cpu_only_sched(size_t graph_size) {
    ggml_backend_t cpu = cpu_backend();
    return ggml_backend_sched_new(&cpu, nullptr, 1, graph_size,
                                      /*parallel=*/false,
                                      /*op_offload=*/false);
}

ggml_backend_sched_t make_sched(size_t graph_size) {
    // Two backends: the default (Vulkan or CPU) and an explicit CPU
    // fallback for ops the primary doesn't support. If the default is
    // already CPU there's no need to register it twice — pass just one.
    ggml_backend_t prim = default_backend();
    if (prim == cpu_backend()) {
        return ggml_backend_sched_new(&prim, nullptr, 1, graph_size,
                                          /*parallel=*/false,
                                          /*op_offload=*/false);
    }
    ggml_backend_t backends[2] = { prim, cpu_backend() };
    return ggml_backend_sched_new(backends, nullptr, 2, graph_size,
                                      /*parallel=*/false,
                                      /*op_offload=*/false);
}

}  // namespace chatterbox
