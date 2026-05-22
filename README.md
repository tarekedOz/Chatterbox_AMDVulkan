# Chatterbox TTS — AMD Vulkan port

A clean-room C++ port of [Chatterbox TTS](https://github.com/resemble-ai/chatterbox)
(Resemble AI, Turbo variant) that runs on **AMD Strix Halo** (Radeon
8060S, `gfx1151`) via the **Vulkan backend of `ggml`** — no ROCm, no CUDA.
The architecture mirrors `whisper.cpp`: a self-contained C++ inference
engine over `ggml` + GGUF, with a thin Rust HTTP server in front exposing
the OpenAI `/v1/audio/speech` contract, shippable as a single Docker image.

A Windows-native binary falls out of the same codebase for free.

## Why

The 8060S iGPU delivers serious compute (~16 TFLOPS fp16) but no
first-class ML framework targets it on Linux. `ggml`'s Vulkan backend is
production-tested across AMD/Nvidia/Intel/Apple, so building on it gets the
iGPU working today *and* yields free portability to other GPUs.

## Status

Full pipeline runs end-to-end (text + reference WAV → 24 kHz PCM) on both
CPU and Vulkan, with every component parity-tested against NumPy/PyTorch
oracles (**21/21 tests pass** on the Vulkan build).

Performance on the 8060S (short utterance, "Hello world." / 30 speech tokens):

| | CPU | Vulkan |
|---|---|---|
| Synthesis | ~12 s | **~0.5 s** |
| End-to-end (incl. one-time model load) | ~13 s | **~2.2 s** |

The Vulkan speedup comes from keeping all ops on the iGPU with per-matmul
fp32 accumulation (to clear fp16 drift), reusing the T3 scheduler +
pre-transposing its weights, and precomputing the mel-extractor DFT tables.

## Repo layout

```
chatterbox-cpp/      C++ inference engine over ggml + GGUF  (see its README)
chatterbox-server/   Rust HTTP server (axum), FFI to the engine's C ABI
scripts/             PyTorch→GGUF conversion + NumPy parity oracles
models/              GGUF weights (gitignored — see below)
tests/               parity reference binaries + reference voices
dist/                Windows installer (thin, download-on-first-run)
Dockerfile           multi-stage build → single Debian-slim image
```

## Install (Windows)

The Windows build is a single self-contained `chatterbox-server.exe`
(engine + web UI + HTTP, Vulkan-accelerated — no Docker, no runtime DLLs).
The installer is **thin**: the ~1.4 GB model weights download + verify on
first launch (like Ollama / LM Studio), they aren't bundled.

**One call** (PowerShell) — downloads the ~8 MB app, fetches + checksums
the weights, adds a Start Menu shortcut, and opens the UI:

```powershell
irm https://github.com/tarekedOz/Chatterbox_AMDVulkan/releases/download/v1/install.ps1 | iex
```

**Or the GUI installer** — download `chatterbox-tts-setup.exe` from the
release and run it (same result).

Either way, first launch does a one-time weights download (cached
afterward), then serves the web UI at **http://127.0.0.1:8087/** with GPU
acceleration via Vulkan. Installer internals + how to build/publish it
(manifest, downloader, Inno Setup script) are in [`dist/`](dist/README.md).

**Upgrade** — no auto-update; re-run the new release's one-liner (each
release's `install.ps1` defaults to its own tag). It re-extracts the app
(your `config.yaml` is preserved) and only re-downloads weights whose
SHA-256 changed; the GUI installer installs in place over the previous
version.

**Uninstall** — GUI install: **Settings → Apps** or the *Uninstall
Chatterbox TTS* Start Menu entry. One-call install: the *Uninstall
Chatterbox TTS* Start Menu shortcut (runs `uninstall.ps1`). Both remove
the downloaded weights. Full details in [`dist/README.md`](dist/README.md#uninstall--upgrade).

> Distribution model: the installer carries only the binary + a manifest
> (file list + SHA-256); `fetch-models.ps1` pulls the weights from the
> release host on first run and verifies each checksum.

## Build & run from source (developers)

**1. Build the engine** — see [`chatterbox-cpp/README.md`](chatterbox-cpp/README.md)
for prerequisites (Scoop GCC/CMake/Ninja/Vulkan), model setup, and the full
build/test commands. In short:

```powershell
git submodule update --init --recursive
$env:CC = "$HOME/scoop/apps/gcc/current/bin/gcc.exe"; $env:CXX = "$HOME/scoop/apps/gcc/current/bin/g++.exe"
cmake -S chatterbox-cpp -B chatterbox-cpp/build_vk -G Ninja -DCMAKE_BUILD_TYPE=Release -DCHATTERBOX_VULKAN=ON
cmake --build chatterbox-cpp/build_vk -j
ctest --test-dir chatterbox-cpp/build_vk --output-on-failure
```

**2. Run the HTTP server + web UI**

```powershell
cd chatterbox-server
# On Windows the build links libgomp from scoop's gcc; put it on PATH and
# point the build at its lib dir (gcc is invoked by full path elsewhere
# and isn't on the persistent PATH):
$env:Path = "$HOME/scoop/apps/gcc/current/bin;$env:Path"
$env:CHATTERBOX_GCC_LIB_DIR = "$HOME/scoop/apps/gcc/current/lib"
cargo build --release
./target/release/chatterbox-server.exe `
    --t3-gguf ../models/chatterbox-turbo-t3-fp16.gguf `
    --ve-gguf ../models/chatterbox-turbo-ve-fp16.gguf `
    --s3gen-gguf ../models/chatterbox-turbo-s3gen-fp16.gguf `
    --voices-gguf ../tests/voices/voices.gguf `
    --addr 127.0.0.1:8087
```

Then open **http://127.0.0.1:8087/** for the web UI (text box, voice
picker, parameters, presets, history, audio player + download,
light/dark theme).

Settings can also come from a YAML config file instead of flags — see
[`chatterbox-server/config.example.yaml`](chatterbox-server/config.example.yaml).
Precedence is **CLI flag > env var > config file > default**; with a
`config.yaml` in the working dir (or `--config <path>`) the server can
start with no flags at all.

Endpoints:

```
# Web UI
GET  /                    -> embedded single-page UI

# Rich API (used by the UI)
GET  /api/voices          -> {"voices": ["Abigail", "Adrian", ...]}
GET  /api/config          -> {"formats": [...], "voices": [...], "max_chunk_chars": N}
POST /api/clone           -> multipart WAV upload; clones the voice for
                             subsequent /api/tts calls (send voice:"")
POST /api/tts             -> audio  {text, voice, seed?, format?,
                                     temperature?, top_p?, top_k?,
                                     repetition_penalty?, cfm_timesteps?,
                                     stream?}
                             voice:"" uses the active clone; omitted params
                             use engine defaults; long text auto-chunks;
                             stream:true -> chunked raw PCM (audio/L16).

# OpenAI-compatible (frozen; drop-in for OpenAI /v1/audio/speech clients)
GET  /health              -> "ok"
GET  /v1/audio/voices     -> {"voices": [...]}
POST /v1/audio/speech     -> WAV / raw PCM   {input, voice, response_format, seed, model, speed}
```

Output formats: `wav` and `pcm` are always available; `mp3` and `opus`
require a build with `--features audio-formats` (on by default in the
Docker image; see below). `GET /api/config` reports what the running
build supports, and the UI dropdown follows it.

**3. Or run the Docker image** (CPU)

```sh
docker build -t chatterbox-tts:latest .
docker run --rm -p 8087:8087 chatterbox-tts:latest
# open http://localhost:8087/ for the UI, or call the API:
curl -X POST localhost:8087/v1/audio/speech \
  -H 'content-type: application/json' \
  -d '{"input":"Hello from Chatterbox.","voice":"Adrian"}' --output out.wav
```

The Docker image bundles the web UI (embedded in the binary), ships the
CPU backend, and builds the MP3/Opus encoders by default
(`--build-arg AUDIO_FORMATS=OFF` for the lean WAV/PCM-only build).
Vulkan-in-container (GPU passthrough via `--device /dev/dri`) is a
follow-up.

## Models

The three GGUFs (~1.3 GB total) are **not** committed (`*.gguf` is
gitignored). Place them in `models/`:

- `chatterbox-turbo-t3-fp16.gguf` — autoregressive backbone
- `chatterbox-turbo-ve-fp16.gguf` — voice encoder
- `chatterbox-turbo-s3gen-fp16.gguf` — speaker enc + flow + CFM + vocoder

Either copy them from an existing build or regenerate from the upstream
Turbo checkpoints with `scripts/convert_*_to_gguf.py` (needs Python +
torch + librosa).

### Voice library

The 28 reference voice clips live in `voices/`. The server/Docker image
use a packed `tests/voices/voices.gguf` (precomputed conditioning, not
committed — it's an artifact). Regenerate it after changing the voice set:

```sh
chatterbox-cpp/build/pack_voices \
    models/chatterbox-turbo-t3-fp16.gguf \
    models/chatterbox-turbo-ve-fp16.gguf \
    models/chatterbox-turbo-s3gen-fp16.gguf \
    voices/  tests/voices/voices.gguf
```

~30 s on the Vulkan build for all 28 voices; the Dockerfile bundles the
resulting `voices.gguf`.

## Docs

- [`chatterbox-cpp/README.md`](chatterbox-cpp/README.md) — engine build/run reference

## Credits

Built on [`ggml`](https://github.com/ggml-org/ggml). Model architecture and
weights from [Chatterbox](https://github.com/resemble-ai/chatterbox)
(Resemble AI, MIT). API contract mirrors
[Chatterbox-TTS-Server](https://github.com/devnen/Chatterbox-TTS-Server).
