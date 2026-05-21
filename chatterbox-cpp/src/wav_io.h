#pragma once

// Minimal WAV reader / writer for the smoke test path. Handles only
// PCM mono/stereo at 8/16/24/32-bit integer or 32-bit float. Stereo
// inputs are downmixed to mono on read (channel average). Multi-chunk
// (RIFF subchunks) is supported.
//
// This is NOT a full WAV parser. It exists so the integration test can
// load a real voice sample without depending on libsndfile.

#include <cstdint>
#include <string>
#include <vector>

namespace chatterbox {

// Read a WAV file into a mono fp32 buffer.
// `out_audio` ends in [-1, 1] approximately; `out_sample_rate` carries
// the source sample rate. Stereo inputs are downmixed by channel-average.
// Returns false on any parse failure.
bool read_wav_mono(const std::string& path,
                     std::vector<float>& out_audio,
                     int& out_sample_rate);

// Write a mono fp32 buffer to a 16-bit PCM WAV.
// fp32 input is clipped to [-1, 1] then scaled to int16.
bool write_wav_mono(const std::string& path,
                      const std::vector<float>& audio,
                      int sample_rate);

}  // namespace chatterbox
