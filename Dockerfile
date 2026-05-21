# Multi-stage build for the Chatterbox TTS server.
#
# Output: a single Debian-slim image with chatterbox-server, the three
# GGUFs, and the voices.gguf bundled at /models/. Listens on 0.0.0.0:8087
# by default.
#
# Build:
#     docker build -t chatterbox-tts:latest .
#
# Run (CPU):
#     docker run --rm -p 8087:8087 chatterbox-tts:latest
#
# Vulkan (WIP, not yet functional — segfaults during compute; see
# session-state.md notes):
#     docker build --build-arg CHATTERBOX_VULKAN=ON -t chatterbox-tts:vk .
#     docker run --rm --device /dev/dri:/dev/dri \
#         -p 8087:8087 chatterbox-tts:vk

ARG CHATTERBOX_VULKAN=OFF
# Build MP3 + Opus encoders into the server (cargo feature
# "audio-formats"). On by default; set --build-arg AUDIO_FORMATS=OFF for
# the lean WAV/PCM-only build. mp3lame vendors+builds libmp3lame from
# source; audiopus links system libopus (libopus-dev below).
ARG AUDIO_FORMATS=ON

# ----- Stage 1: C++ engine + Rust server build -----
FROM rust:1.95-bookworm AS builder

ARG CHATTERBOX_VULKAN
ARG AUDIO_FORMATS

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake ninja-build pkg-config \
        $( [ "$CHATTERBOX_VULKAN" = "ON" ] && echo libvulkan-dev glslc ) \
        $( [ "$AUDIO_FORMATS" = "ON" ] && echo libopus-dev ) \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY chatterbox-cpp chatterbox-cpp
COPY chatterbox-server chatterbox-server

# C++ engine (the static libs the Rust server links against).
# BUILD_SHARED_LIBS=OFF is required: ggml's CMake sees a Linux toolchain
# and would otherwise build .so files, but GGML_STATIC=ON (forced in our
# CMakeLists) adds `-static` to the linker flags. The two together produce
# `-static -shared` which fails with the "failed to set dynamic section
# sizes" linker error.
WORKDIR /workspace/chatterbox-cpp
RUN cmake -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCHATTERBOX_BUILD_TESTS=OFF \
        -DBUILD_SHARED_LIBS=OFF \
        -DCHATTERBOX_VULKAN=${CHATTERBOX_VULKAN} \
 && cmake --build build --target chatterbox -j

# Rust server.
WORKDIR /workspace/chatterbox-server
ENV CHATTERBOX_CPP_BUILD_DIR=/workspace/chatterbox-cpp/build
RUN cargo build --release \
        $( [ "$AUDIO_FORMATS" = "ON" ] && echo --features audio-formats )


# ----- Stage 2: runtime -----
FROM debian:bookworm-slim AS runtime

ARG CHATTERBOX_VULKAN
ARG AUDIO_FORMATS

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        $( [ "$CHATTERBOX_VULKAN" = "ON" ] && echo libvulkan1 mesa-vulkan-drivers ) \
        $( [ "$AUDIO_FORMATS" = "ON" ] && echo libopus0 ) \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /workspace/chatterbox-server/target/release/chatterbox-server \
    /usr/local/bin/chatterbox-server

# Models bundled at /models. Override with `-v $PWD/models:/models` to
# use a different set.
COPY models/chatterbox-turbo-t3-fp16.gguf       /models/t3.gguf
COPY models/chatterbox-turbo-ve-fp16.gguf       /models/ve.gguf
COPY models/chatterbox-turbo-s3gen-fp16.gguf    /models/s3gen.gguf
COPY tests/voices/voices.gguf                    /models/voices.gguf

ENV CHATTERBOX_T3_GGUF=/models/t3.gguf      \
    CHATTERBOX_VE_GGUF=/models/ve.gguf      \
    CHATTERBOX_S3GEN_GGUF=/models/s3gen.gguf \
    CHATTERBOX_VOICES_GGUF=/models/voices.gguf \
    CHATTERBOX_ADDR=0.0.0.0:8087

EXPOSE 8087

ENTRYPOINT ["/bin/sh", "-c", "\
    chatterbox-server \
        --t3-gguf $CHATTERBOX_T3_GGUF \
        --ve-gguf $CHATTERBOX_VE_GGUF \
        --s3gen-gguf $CHATTERBOX_S3GEN_GGUF \
        --voices-gguf $CHATTERBOX_VOICES_GGUF \
        --addr $CHATTERBOX_ADDR \
"]
