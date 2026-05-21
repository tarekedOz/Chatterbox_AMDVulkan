cmake_minimum_required(VERSION 3.21)
project(chatterbox-cpp
    LANGUAGES CXX C
    VERSION 0.1.0
    DESCRIPTION "Chatterbox TTS inference on ggml (Vulkan-target)"
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Lavavej mingw-w64 (scoop gcc) ships older Win11-SDK headers that lack
# THREAD_POWER_THROTTLING_STATE, which ggml-cpu's thread-priority code
# guards behind _WIN32_WINNT >= 0x0602. Force the macro to Win7 on this
# toolchain so the block is preprocessed out. The lost optimization only
# matters on Win11 anyway; ggml falls back to plain SetThreadPriority.
if(WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    add_compile_definitions(_WIN32_WINNT=0x0601 WINVER=0x0601)
endif()

# --------------------------------------------------------------------------
# Backend options. Phase 1.E stays CPU-only; Phase 2 flips CHATTERBOX_VULKAN.
# --------------------------------------------------------------------------
option(CHATTERBOX_VULKAN "Enable Vulkan backend via ggml" OFF)
option(CHATTERBOX_BUILD_TESTS "Build chatterbox-cpp tests" ON)

# Forward to ggml's own options before adding its subdir. ggml's CMake reads
# these as cache vars, so set them with FORCE to override any prior values.
set(GGML_VULKAN          ${CHATTERBOX_VULKAN} CACHE BOOL "" FORCE)
set(GGML_NATIVE          OFF                  CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS     OFF                  CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES  OFF                  CACHE BOOL "" FORCE)
set(GGML_STATIC          ON                   CACHE BOOL "" FORCE)

# Bump ggml's per-tensor name length cap. Default 64 chars is too tight for
# our identity-named S3Gen tensors (some are 72 chars deep, e.g.
# speaker_encoder.xvector.block1.tdnnd10.nonlinear1.batchnorm.running_mean).
# Cost: ~64 extra bytes per ggml_tensor struct, ~160 KiB for S3Gen's 2491
# tensors. Negligible. Must match between writer (python gguf) and reader,
# which is automatic for us since both pin v0.12.0.
add_compile_definitions(GGML_MAX_NAME=128)

# Surface the Vulkan toggle to the engine source via a preprocessor
# define so src/backend.cpp picks up the Vulkan init path.
if(CHATTERBOX_VULKAN)
    add_compile_definitions(GGML_USE_VULKAN)
endif()

add_subdirectory(third_party/ggml EXCLUDE_FROM_ALL)

# --------------------------------------------------------------------------
# chatterbox library — internal C++ engine
# --------------------------------------------------------------------------
add_library(chatterbox STATIC
    src/model.cpp
    src/tokenizer.cpp
    src/t3.cpp
    src/sampler.cpp
    src/ve.cpp
    src/mel.cpp
    src/s3enc.cpp
    src/s3tok.cpp
    src/s3spk.cpp
    src/s3gen_mel.cpp
    src/flow_encoder.cpp
    src/cfm_decoder.cpp
    src/hift_vocoder.cpp
    src/resample.cpp
    src/kaldi_fbank.cpp
    src/ve_mel.cpp
    src/chatterbox.cpp
    src/wav_io.cpp
    src/voice_pack.cpp
    src/backend.cpp
    src/chatterbox_capi.cpp
)
target_include_directories(chatterbox
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/src
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(chatterbox PUBLIC ggml)
target_compile_features(chatterbox PUBLIC cxx_std_17)

# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------
if(CHATTERBOX_BUILD_TESTS)
    enable_testing()

    # Paths to the GGUFs produced by scripts/convert_*_to_gguf.py.
    # Override on the command line if needed (-DCHATTERBOX_T3_GGUF=...).
    set(CHATTERBOX_T3_GGUF
        "${CMAKE_CURRENT_SOURCE_DIR}/../models/chatterbox-turbo-t3-fp16.gguf"
        CACHE FILEPATH "T3 GGUF used by smoke tests")
    set(CHATTERBOX_VE_GGUF
        "${CMAKE_CURRENT_SOURCE_DIR}/../models/chatterbox-turbo-ve-fp16.gguf"
        CACHE FILEPATH "VE GGUF used by smoke tests")
    set(CHATTERBOX_S3GEN_GGUF
        "${CMAKE_CURRENT_SOURCE_DIR}/../models/chatterbox-turbo-s3gen-fp16.gguf"
        CACHE FILEPATH "S3Gen GGUF used by smoke tests")

    set(CHATTERBOX_TOKENIZER_GROUNDTRUTH
        "${CMAKE_CURRENT_SOURCE_DIR}/../tests/tokenizer_groundtruth.json"
        CACHE FILEPATH "Python-side oracle for the tokenizer parity test")
    set(CHATTERBOX_T3_REFERENCE
        "${CMAKE_CURRENT_SOURCE_DIR}/../tests/t3_reference.bin"
        CACHE FILEPATH "NumPy reference binary for the T3 forward parity test")

    add_executable(load_gguf_smoke tests/load_gguf_smoke.cpp)
    target_link_libraries(load_gguf_smoke PRIVATE chatterbox)

    add_executable(tokenizer_test tests/tokenizer_test.cpp)
    target_link_libraries(tokenizer_test PRIVATE chatterbox)

    add_executable(t3_forward_test tests/t3_forward_test.cpp)
    target_link_libraries(t3_forward_test PRIVATE chatterbox)

    add_executable(sampler_test tests/sampler_test.cpp)
    target_link_libraries(sampler_test PRIVATE chatterbox)

    add_executable(t3_generate_smoke tests/t3_generate_smoke.cpp)
    target_link_libraries(t3_generate_smoke PRIVATE chatterbox)

    add_executable(ve_forward_test tests/ve_forward_test.cpp)
    target_link_libraries(ve_forward_test PRIVATE chatterbox)

    add_executable(mel_test tests/mel_test.cpp)
    target_link_libraries(mel_test PRIVATE chatterbox)

    add_executable(s3_encoder_test tests/s3_encoder_test.cpp)
    target_link_libraries(s3_encoder_test PRIVATE chatterbox)

    add_executable(s3_tokenizer_test tests/s3_tokenizer_test.cpp)
    target_link_libraries(s3_tokenizer_test PRIVATE chatterbox)

    add_executable(pipeline_smoke tests/pipeline_smoke.cpp)
    target_link_libraries(pipeline_smoke PRIVATE chatterbox)

    add_executable(campplus_test tests/campplus_test.cpp)
    target_link_libraries(campplus_test PRIVATE chatterbox)

    add_executable(campplus_fcm_debug tests/campplus_fcm_debug.cpp)
    target_link_libraries(campplus_fcm_debug PRIVATE chatterbox)

    add_executable(s3gen_mel_test tests/s3gen_mel_test.cpp)
    target_link_libraries(s3gen_mel_test PRIVATE chatterbox)

    add_executable(flow_encoder_test tests/flow_encoder_test.cpp)
    target_link_libraries(flow_encoder_test PRIVATE chatterbox)

    add_executable(cfm_decoder_test tests/cfm_decoder_test.cpp)
    target_link_libraries(cfm_decoder_test PRIVATE chatterbox)

    add_executable(hift_vocoder_test tests/hift_vocoder_test.cpp)
    target_link_libraries(hift_vocoder_test PRIVATE chatterbox)

    add_executable(resample_test tests/resample_test.cpp)
    target_link_libraries(resample_test PRIVATE chatterbox)

    add_executable(kaldi_fbank_test tests/kaldi_fbank_test.cpp)
    target_link_libraries(kaldi_fbank_test PRIVATE chatterbox)

    add_executable(ve_mel_test tests/ve_mel_test.cpp)
    target_link_libraries(ve_mel_test PRIVATE chatterbox)

    add_executable(chatterbox_smoke tests/chatterbox_smoke.cpp)
    target_link_libraries(chatterbox_smoke PRIVATE chatterbox)

    add_executable(chatterbox_real_voice tests/chatterbox_real_voice.cpp)
    target_link_libraries(chatterbox_real_voice PRIVATE chatterbox)

    add_executable(voice_pack_test tests/voice_pack_test.cpp)
    target_link_libraries(voice_pack_test PRIVATE chatterbox)
endif()

# Tools
add_executable(pack_voices tools/pack_voices.cpp)
target_link_libraries(pack_voices PRIVATE chatterbox)

if(CHATTERBOX_BUILD_TESTS)

    add_test(NAME load_gguf_smoke_t3
             COMMAND load_gguf_smoke "${CHATTERBOX_T3_GGUF}")
    add_test(NAME load_gguf_smoke_ve
             COMMAND load_gguf_smoke "${CHATTERBOX_VE_GGUF}")
    add_test(NAME load_gguf_smoke_s3gen
             COMMAND load_gguf_smoke "${CHATTERBOX_S3GEN_GGUF}")
    add_test(NAME tokenizer_parity
             COMMAND tokenizer_test
                     "${CHATTERBOX_T3_GGUF}"
                     "${CHATTERBOX_TOKENIZER_GROUNDTRUTH}")
    add_test(NAME t3_forward_parity
             COMMAND t3_forward_test
                     "${CHATTERBOX_T3_GGUF}"
                     "${CHATTERBOX_T3_REFERENCE}")
    add_test(NAME sampler_unit
             COMMAND sampler_test)
    add_test(NAME t3_generate_smoke
             COMMAND t3_generate_smoke
                     "${CHATTERBOX_T3_GGUF}"
                     "${CHATTERBOX_T3_REFERENCE}")
    add_test(NAME ve_forward_parity
             COMMAND ve_forward_test
                     "${CHATTERBOX_VE_GGUF}"
                     "${CMAKE_CURRENT_SOURCE_DIR}/../tests/ve_reference.bin")
    add_test(NAME mel_parity
             COMMAND mel_test
                     "${CHATTERBOX_S3GEN_GGUF}"
                     "${CMAKE_CURRENT_SOURCE_DIR}/../tests/s3_mel_reference.bin")
    add_test(NAME s3_encoder_parity
             COMMAND s3_encoder_test
                     "${CHATTERBOX_S3GEN_GGUF}"
                     "${CMAKE_CURRENT_SOURCE_DIR}/../tests/s3_encoder_reference.bin")
    add_test(NAME s3_tokenizer_parity
             COMMAND s3_tokenizer_test
                     "${CHATTERBOX_S3GEN_GGUF}"
                     "${CMAKE_CURRENT_SOURCE_DIR}/../tests/s3_tokenizer_reference.bin")
    add_test(NAME pipeline_smoke
             COMMAND pipeline_smoke
                     "${CHATTERBOX_T3_GGUF}"
                     "${CHATTERBOX_VE_GGUF}"
                     "${CHATTERBOX_S3GEN_GGUF}"
                     "${CMAKE_CURRENT_SOURCE_DIR}/../tests/ve_reference.bin"
                     "${CMAKE_CURRENT_SOURCE_DIR}/../tests/s3_tokenizer_reference.bin")
    add_test(NAME campplus_parity
             COMMAND campplus_test
                     "${CHATTERBOX_S3GEN_GGUF}"
                     "${CMAKE_CURRENT_SOURCE_DIR}/../tests/campplus_reference.bin")
    add_test(NAME s3gen_mel_parity
             COMMAND s3gen_mel_test
                     "${CMAKE_CURRENT_SOURCE_DIR}/../tests/s3gen_mel_filters.bin"
                     "${CMAKE_CURRENT_SOURCE_DIR}/../tests/s3gen_mel_reference.bin")
    add_test(NAME flow_encoder_parity
             COMMAND flow_encoder_test
                     "${CHATTERBOX_S3GEN_GGUF}"
                     "${CMAKE_CURRENT_SOURCE_DIR}/../tests/flow_encoder_reference.bin")
    add_test(NAME cfm_decoder_parity
             COMMAND cfm_decoder_test
                     "${CHATTERBOX_S3GEN_GGUF}"
                     "${CMAKE_CURRENT_SOURCE_DIR}/../tests/cfm_decoder_reference.bin")
    add_test(NAME hift_vocoder_parity
             COMMAND hift_vocoder_test
                     "${CHATTERBOX_S3GEN_GGUF}"
                     "${CMAKE_CURRENT_SOURCE_DIR}/../tests/hift_vocoder_reference.bin")
    add_test(NAME resample_parity
             COMMAND resample_test
                     "${CMAKE_CURRENT_SOURCE_DIR}/../tests/resampler_reference.bin")
    add_test(NAME kaldi_fbank_parity
             COMMAND kaldi_fbank_test
                     "${CMAKE_CURRENT_SOURCE_DIR}/../tests/kaldi_fbank_reference.bin")
    add_test(NAME ve_mel_parity
             COMMAND ve_mel_test
                     "${CMAKE_CURRENT_SOURCE_DIR}/../tests/ve_mel_reference.bin")
    add_test(NAME chatterbox_smoke
             COMMAND chatterbox_smoke
                     "${CHATTERBOX_T3_GGUF}"
                     "${CHATTERBOX_VE_GGUF}"
                     "${CHATTERBOX_S3GEN_GGUF}")
endif()
