#pragma once

// Backend selection. The C++ engine uses a single process-wide ggml
// backend for both model weights (allocated in backend memory at
// load_model time) and graph compute. At build time we pick either
// CPU-only (default) or Vulkan-on-CPU-fallback via the
// CHATTERBOX_VULKAN cmake option (-> defines GGML_USE_VULKAN).
//
// `default_backend()` returns a Vulkan backend when one is available
// and CHATTERBOX_DISABLE_VULKAN is not set, otherwise CPU. The
// returned pointer is a NON-OWNING reference to a process-wide
// singleton — callers must NOT call ggml_backend_free on it.
//
// `default_buft()` returns the matching ggml_backend_buffer_type_t.

#include "ggml-backend.h"

#include <vector>

struct ggml_tensor;

namespace chatterbox {

// Process-wide singleton backend. Initialized on first call; never
// freed during the process lifetime.
ggml_backend_t             default_backend();
ggml_backend_buffer_type_t default_buft();

// CPU fallback backend (always available, also a process-wide
// singleton). Used as the second backend in the ggml_backend_sched_t
// so unsupported Vulkan ops fall back to CPU automatically.
ggml_backend_t             cpu_backend();

// Build a fresh ggml_backend_sched_t whose primary backend is
// `default_backend()` and whose fallback is `cpu_backend()`. Caller
// owns the returned sched and must free it with
// ggml_backend_sched_free. Returns nullptr on failure.
//
// `graph_size` should be at least the max number of nodes the graph
// will contain. The sched preallocates internal tables of this size.
ggml_backend_sched_t       make_sched(size_t graph_size);

// Build a CPU-ONLY sched. Used for components whose discrete output
// (e.g. FSQ tokenization) is fp16-noise-sensitive: routing them through
// CPU keeps the fp32 accumulation that the parity references were
// captured against. Model weights remain in their original backend
// buffer; sched copies them to a CPU-visible buffer for compute. On
// UMA systems (Strix Halo iGPU) this is effectively zero-copy.
ggml_backend_sched_t       make_cpu_only_sched(size_t graph_size);

bool is_vulkan_backend(ggml_backend_t backend);
bool is_vulkan_active();      // convenience: is_vulkan_backend(default_backend())

// Download a ggml tensor's data into a host-side fp32 std::vector. Works
// regardless of which backend owns the tensor — uses
// ggml_backend_tensor_get under the hood. Handles fp16 -> fp32
// conversion on the host side. Returns an empty vector for any other
// dtype.
std::vector<float> read_tensor_f32(const ggml_tensor* t);

}  // namespace chatterbox
