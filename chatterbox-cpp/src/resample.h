#pragma once

// Host-side audio resampler. Polyphase upsample/FIR/downsample with a
// Kaiser-windowed sinc anti-alias filter. Matches
// scipy.signal.resample_poly(x, up, down, window=('kaiser', 8.6)) within
// fp32 numerical drift; see scripts/reference_resampler.py.
//
// Used by the conditioning path of `chatterbox::Chatterbox::synthesize`
// to convert the reference WAV from whatever sample rate the user
// provides to:
//   - 16 kHz for the S3Tokenizer + CAMPPlus path
//   - 24 kHz for the S3Gen mel extractor
//
// Single-channel input; multi-channel inputs should be downmixed by the
// caller before passing in.

#include <cstdint>
#include <vector>

namespace chatterbox {

// Resample a mono PCM buffer from `in_sr` to `out_sr`. Returns an empty
// vector on invalid inputs.
//
// Output length: ceil(in.size() * out_sr / in_sr) rounded by the
// polyphase output formula -- matches scipy's resample_poly length.
std::vector<float> resample_audio(const std::vector<float>& in,
                                    int in_sr, int out_sr);

}  // namespace chatterbox
