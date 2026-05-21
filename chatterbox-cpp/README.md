# chatterbox-cpp

Clean-room C++ inference engine for [Chatterbox TTS](https://github.com/resemble-ai/chatterbox)
(Turbo variant) over `ggml` + GGUF. Architecture mirrors `whisper.cpp`: a
self-contained C++ engine with a small C ABI, CPU or Vulkan backend, no
Python at runtime.

The full text-to-speech pipeline runs end-to-end: text + a reference WAV
in, 24 kHz PCM out. On an AMD Strix Halo (Radeon 8060S, `gfx1151`) iGPU
via Vulkan, synthesis of a short utterance is ~0.5 s (vs ~13 s on CPU).

## Pipeline

```
text  ──tokenizer──▶ text ids ─┐
                               ├─▶ T3 (24-layer AR transformer, KV cache)
ref WAV ─┬─VE mel─▶ VE LSTM ───┘        │  speech tokens
         │                              ▼
         ├─Kaldi fbank─▶ CAMPPlus ─▶ spk emb ─▶ FlowEncoder (conformer)
         ├─S3 mel─▶ S3Tokenizer ─▶ prompt ids        │  mel features
         └─S3Gen mel ─▶ prompt feat                  ▼
                                          CFM decoder (meanflow solver)
                                                     │  mel
                                                     ▼
                                          HiFTGenerator vocoder ─▶ 24 kHz PCM
```

8 model components + 4 host-side mel/feature extractors, all parity-tested
against NumPy/PyTorch oracles.

## Layout

```
chatterbox-cpp/
├── CMakeLists.txt
├── third_party/ggml/        submodule, pinned at v0.12.0
├── src/                     engine (model loader, T3, VE, S3Gen stack,
│                            vocoder, host extractors, orchestration)
│   ├── chatterbox.{h,cpp}   top-level orchestration (condition + synthesize)
│   ├── backend.{h,cpp}      CPU/Vulkan backend selection + sched helpers
│   └── chatterbox_capi.h    C ABI consumed by the Rust server
├── tools/pack_voices.cpp    bake N reference voices into a voices GGUF
└── tests/                   21 parity + smoke tests (ctest)
```

## Prerequisites (Windows)

Built with the mingw-w64 GCC toolchain + Ninja (not MSVC). Install via
[Scoop](https://scoop.sh):

```powershell
scoop install git gcc cmake ninja
scoop install vulkan        # provides glslc.exe, the shader compiler ggml needs
scoop install rust          # only for the HTTP server (chatterbox-server)
```

For the Vulkan backend you also need a working Vulkan driver (ICD). On
Strix Halo that's the **AMD Adrenalin driver** — confirm with `vulkaninfo`.

Linux works too: a recent GCC/Clang, CMake, Ninja, and the Vulkan SDK +
`mesa-vulkan-drivers` (RADV) for `gfx1151`.

## Models

The three GGUFs are **not** committed (`*.gguf` is gitignored). Place them
in `../models/` relative to this directory:

| File | Size | Arch |
|---|---|---|
| `chatterbox-turbo-t3-fp16.gguf`    | 818 MB | `chatterbox_t3` |
| `chatterbox-turbo-ve-fp16.gguf`    | 2.8 MB | `chatterbox_ve` |
| `chatterbox-turbo-s3gen-fp16.gguf` | 509 MB | `chatterbox_s3gen` |

Either copy them from an existing build, or regenerate from the upstream
Turbo checkpoints with the conversion scripts (needs Python + torch +
librosa, see `../scripts/`):

```sh
python ../scripts/convert_chatterbox_to_gguf.py <checkpoints> --out ../models/chatterbox-turbo-t3-fp16.gguf
python ../scripts/convert_ve_to_gguf.py          <checkpoints> --out ../models/chatterbox-turbo-ve-fp16.gguf
python ../scripts/convert_s3gen_to_gguf.py       <checkpoints> --out ../models/chatterbox-turbo-s3gen-fp16.gguf
```

## Build

Init the ggml submodule once, then configure + build. The compiler must be
pointed at Scoop's GCC.

```powershell
git submodule update --init --recursive

$env:CC  = "$HOME/scoop/apps/gcc/current/bin/gcc.exe"
$env:CXX = "$HOME/scoop/apps/gcc/current/bin/g++.exe"

# Vulkan build:
cmake -S . -B build_vk -G Ninja -DCMAKE_BUILD_TYPE=Release -DCHATTERBOX_VULKAN=ON
cmake --build build_vk -j

# CPU-only build (separate dir, drop the Vulkan flag):
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

CMake options: `-DCHATTERBOX_VULKAN=ON|OFF` (default OFF),
`-DCHATTERBOX_BUILD_TESTS=ON|OFF` (default ON). The GGUF paths used by
tests can be overridden with `-DCHATTERBOX_T3_GGUF=...` etc.

## Verify

```powershell
ctest --test-dir build_vk --output-on-failure        # expect 21/21 pass

# end-to-end synthesis smoke (synthetic reference WAV, "Hello world."):
cd build_vk
./chatterbox_smoke.exe ../../models/chatterbox-turbo-t3-fp16.gguf `
                       ../../models/chatterbox-turbo-ve-fp16.gguf `
                       ../../models/chatterbox-turbo-s3gen-fp16.gguf
```

## Runtime env vars

| Var | Effect |
|---|---|
| `CHATTERBOX_DISABLE_VULKAN=1` | Force the CPU backend even in a Vulkan build |
| `CHATTERBOX_PROFILE=1`        | Print per-stage pipeline timings (incl. `cond_*` sub-steps) |
| `CHATTERBOX_T3_PROFILE=1`     | Print T3 per-AR-step breakdown (build/sched/compute/readback/sample) |
| `CHATTERBOX_CFM_PIN_STAGE=<stage>` | Diagnostic: route a CFM stage output through CPU (parity fallback) |

## Backends & numerics

- **CPU** — always available; the reference for parity.
- **Vulkan** — `-DCHATTERBOX_VULKAN=ON`. All ops run on the iGPU. fp16
  matmul accumulation drifts across deep stacks, so the FlowEncoder, CFM
  decoder, and HiFT vocoder pin their matmuls to **fp32 accumulation**
  (`ggml_mul_mat_set_prec(GGML_PREC_F32)`) — correct *and* faster than the
  earlier CPU-pinning approach. T3 keeps a widened tolerance (its KV cache
  is buffer-pinned); top-5 token ids match the fp32 reference.

## C ABI

`src/chatterbox_capi.h` exposes a small C interface (consumed by the
`../chatterbox-server` Rust crate via static linking):

```c
chatterbox_ctx_t* chatterbox_init(t3_path, ve_path, s3gen_path);
int   chatterbox_load_voices(ctx, voices_gguf_path);
int   chatterbox_voice_count(ctx);
int   chatterbox_voice_name(ctx, index, out, max_len);
int   chatterbox_set_voice(ctx, voice_name);
// Clone a voice from mono fp32 PCM at sr Hz (engine resamples); caches it
// as the active conditioning until set_voice/condition_pcm is called again.
int   chatterbox_condition_pcm(ctx, samples, n_samples, sr);
// fills caller's out_pcm[max_samples], returns sample count (or <0); seed/
// max_tokens 0 = defaults; writes 24000 to *out_sample_rate.
int   chatterbox_synthesize(ctx, text, seed, max_tokens,
                            out_pcm, max_samples, out_sample_rate);
// Same, with full per-request params (chatterbox_gen_params_t*: temperature,
// top_k, top_p, repetition_penalty, cfm_timesteps, seed, max_tokens; the
// exaggeration/cfg_weight fields are accepted but ignored — Turbo doesn't
// support them). params may be NULL for all defaults.
int   chatterbox_synthesize_ex(ctx, text, params, out_pcm, max_samples,
                               out_sample_rate);
const char* chatterbox_last_error(void);
void  chatterbox_free(ctx);
```

## Tools

`pack_voices` runs the four conditioning extractors once per reference WAV
and serializes the result into a voices GGUF, so the server applies cached
conditioning in <1 ms instead of re-running the ~conditioning path per
request.

See `../docs/session-state.md` for the full implementation history,
component-by-component parity numbers, and the Vulkan performance work.
