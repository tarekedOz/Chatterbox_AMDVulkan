#include "wav_io.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>

namespace chatterbox {

namespace {

uint32_t read_u32_le(const uint8_t* p) {
    return static_cast<uint32_t>(p[0])
         | (static_cast<uint32_t>(p[1]) << 8)
         | (static_cast<uint32_t>(p[2]) << 16)
         | (static_cast<uint32_t>(p[3]) << 24);
}
uint16_t read_u16_le(const uint8_t* p) {
    return static_cast<uint16_t>(p[0])
         | (static_cast<uint16_t>(p[1]) << 8);
}
void write_u32_le(uint8_t* p, uint32_t v) {
    p[0] = static_cast<uint8_t>(v & 0xFF);
    p[1] = static_cast<uint8_t>((v >> 8) & 0xFF);
    p[2] = static_cast<uint8_t>((v >> 16) & 0xFF);
    p[3] = static_cast<uint8_t>((v >> 24) & 0xFF);
}
void write_u16_le(uint8_t* p, uint16_t v) {
    p[0] = static_cast<uint8_t>(v & 0xFF);
    p[1] = static_cast<uint8_t>((v >> 8) & 0xFF);
}

}  // namespace

bool read_wav_mono(const std::string& path, std::vector<float>& out,
                     int& out_sr) {
    out.clear();
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "read_wav: cannot open %s\n", path.c_str()); return false; }

    uint8_t hdr[12];
    f.read(reinterpret_cast<char*>(hdr), 12);
    if (!f || std::memcmp(hdr, "RIFF", 4) != 0
        || std::memcmp(hdr + 8, "WAVE", 4) != 0) {
        std::fprintf(stderr, "read_wav: not a WAVE file\n");
        return false;
    }

    uint16_t fmt_code = 0, channels = 0, bits = 0;
    uint32_t sr = 0;
    std::vector<uint8_t> data_bytes;

    uint8_t chunk[8];
    while (f.read(reinterpret_cast<char*>(chunk), 8)) {
        const uint32_t size = read_u32_le(chunk + 4);
        if (std::memcmp(chunk, "fmt ", 4) == 0) {
            std::vector<uint8_t> b(size);
            f.read(reinterpret_cast<char*>(b.data()), size);
            if (size < 16) {
                std::fprintf(stderr, "read_wav: fmt chunk too small\n");
                return false;
            }
            fmt_code = read_u16_le(b.data());
            channels = read_u16_le(b.data() + 2);
            sr       = read_u32_le(b.data() + 4);
            bits     = read_u16_le(b.data() + 14);
        } else if (std::memcmp(chunk, "data", 4) == 0) {
            data_bytes.resize(size);
            f.read(reinterpret_cast<char*>(data_bytes.data()), size);
            break;
        } else {
            // Skip unknown chunks.
            f.seekg(size, std::ios::cur);
            // RIFF chunks pad to even sizes.
            if (size & 1) f.seekg(1, std::ios::cur);
        }
    }

    if (sr == 0 || channels == 0 || data_bytes.empty()) {
        std::fprintf(stderr, "read_wav: missing fmt or data chunk\n");
        return false;
    }
    out_sr = static_cast<int>(sr);

    const int n_samples = static_cast<int>(data_bytes.size())
                            / (bits / 8) / channels;
    out.resize(n_samples);

    if (fmt_code == 1) {        // PCM integer
        if (bits == 16) {
            const int16_t* src = reinterpret_cast<const int16_t*>(data_bytes.data());
            for (int i = 0; i < n_samples; ++i) {
                float s = 0.0f;
                for (int c = 0; c < channels; ++c) {
                    s += static_cast<float>(src[i * channels + c]);
                }
                out[i] = s / (channels * 32768.0f);
            }
        } else if (bits == 24) {
            const uint8_t* src = data_bytes.data();
            for (int i = 0; i < n_samples; ++i) {
                float s = 0.0f;
                for (int c = 0; c < channels; ++c) {
                    const uint8_t* p = src + (i * channels + c) * 3;
                    int32_t v = (static_cast<int32_t>(p[0]))
                              | (static_cast<int32_t>(p[1]) << 8)
                              | (static_cast<int32_t>(p[2]) << 16);
                    // Sign-extend from 24 bits.
                    if (v & 0x800000) v |= static_cast<int32_t>(0xFF000000);
                    s += static_cast<float>(v);
                }
                out[i] = s / (channels * 8388608.0f);
            }
        } else if (bits == 32) {
            const int32_t* src = reinterpret_cast<const int32_t*>(data_bytes.data());
            for (int i = 0; i < n_samples; ++i) {
                float s = 0.0f;
                for (int c = 0; c < channels; ++c) {
                    s += static_cast<float>(src[i * channels + c]);
                }
                out[i] = s / (channels * 2147483648.0f);
            }
        } else if (bits == 8) {
            // 8-bit PCM is unsigned.
            const uint8_t* src = data_bytes.data();
            for (int i = 0; i < n_samples; ++i) {
                float s = 0.0f;
                for (int c = 0; c < channels; ++c) {
                    s += static_cast<float>(src[i * channels + c]) - 128.0f;
                }
                out[i] = s / (channels * 128.0f);
            }
        } else {
            std::fprintf(stderr, "read_wav: unsupported bit depth %u\n", bits);
            return false;
        }
    } else if (fmt_code == 3) {        // IEEE float
        if (bits != 32) {
            std::fprintf(stderr, "read_wav: float WAV with bits=%u (need 32)\n", bits);
            return false;
        }
        const float* src = reinterpret_cast<const float*>(data_bytes.data());
        for (int i = 0; i < n_samples; ++i) {
            float s = 0.0f;
            for (int c = 0; c < channels; ++c) s += src[i * channels + c];
            out[i] = s / channels;
        }
    } else {
        std::fprintf(stderr, "read_wav: unsupported fmt code %u\n", fmt_code);
        return false;
    }
    return true;
}

bool write_wav_mono(const std::string& path,
                      const std::vector<float>& audio, int sr) {
    std::ofstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "write_wav: cannot open %s\n", path.c_str()); return false; }

    const uint32_t n_samples = static_cast<uint32_t>(audio.size());
    const uint32_t data_size = n_samples * 2;
    const uint32_t riff_size = 36 + data_size;
    const uint32_t byte_rate = static_cast<uint32_t>(sr) * 2;

    uint8_t hdr[44];
    std::memcpy(hdr + 0, "RIFF", 4);
    write_u32_le(hdr + 4, riff_size);
    std::memcpy(hdr + 8, "WAVE", 4);
    std::memcpy(hdr + 12, "fmt ", 4);
    write_u32_le(hdr + 16, 16);                 // fmt chunk size
    write_u16_le(hdr + 20, 1);                  // PCM
    write_u16_le(hdr + 22, 1);                  // mono
    write_u32_le(hdr + 24, static_cast<uint32_t>(sr));
    write_u32_le(hdr + 28, byte_rate);
    write_u16_le(hdr + 32, 2);                  // block align
    write_u16_le(hdr + 34, 16);                 // bits per sample
    std::memcpy(hdr + 36, "data", 4);
    write_u32_le(hdr + 40, data_size);
    f.write(reinterpret_cast<const char*>(hdr), 44);

    std::vector<int16_t> samples(n_samples);
    for (uint32_t i = 0; i < n_samples; ++i) {
        float s = std::clamp(audio[i], -1.0f, 1.0f);
        samples[i] = static_cast<int16_t>(std::lround(s * 32767.0f));
    }
    f.write(reinterpret_cast<const char*>(samples.data()),
              static_cast<std::streamsize>(samples.size()) * 2);
    return static_cast<bool>(f);
}

}  // namespace chatterbox
