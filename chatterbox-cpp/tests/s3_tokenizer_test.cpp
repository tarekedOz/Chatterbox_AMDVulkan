// S3 audio tokenizer full-pipeline parity test.
//
// Reads tests/s3_tokenizer_reference.bin (produced by
// scripts/reference_s3_tokenizer.py), runs S3Tokenizer::encode against
// the embedded synthetic waveform, and asserts the resulting token id
// sequence matches the NumPy oracle byte-for-byte.
//
// FSQ rounding makes this a strict bit-equality check on the integer
// output, modulo any token-boundary cases where fp16 encoder drift
// happens to push a quantized dim across an integer boundary. The
// 0.999 contraction in the FSQ encode mitigates those at ±1; values
// near 0.5 in any of the 8 projected dims are the most fragile.

#include "model.h"
#include "s3tok.h"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr,
                     "Usage: %s <s3gen.gguf> <s3_tokenizer_reference.bin>\n",
                     argv[0]);
        return 2;
    }

    auto m = chatterbox::load_model(argv[1]);
    if (!m) return 1;

    auto tok = chatterbox::S3Tokenizer::load(m.get());
    if (!tok) {
        std::fprintf(stderr, "Failed to construct S3Tokenizer.\n");
        return 1;
    }
    std::printf("S3Tokenizer loaded.\n");

    std::ifstream in(argv[2], std::ios::binary);
    if (!in) {
        std::fprintf(stderr, "Cannot open %s\n", argv[2]);
        return 1;
    }
    int32_t n_samples = 0;
    in.read(reinterpret_cast<char*>(&n_samples), sizeof(n_samples));
    std::vector<float> audio(n_samples);
    in.read(reinterpret_cast<char*>(audio.data()),
            n_samples * sizeof(float));
    int32_t T_tok = 0;
    in.read(reinterpret_cast<char*>(&T_tok), sizeof(T_tok));
    std::vector<int32_t> expected(T_tok);
    in.read(reinterpret_cast<char*>(expected.data()),
            T_tok * sizeof(int32_t));
    if (!in) {
        std::fprintf(stderr, "Reference file too short.\n");
        return 1;
    }
    std::printf("Reference: %d samples (%.2f s), %d expected tokens\n",
                n_samples, static_cast<float>(n_samples) / chatterbox::S3Tokenizer::SR,
                T_tok);

    auto actual = tok->encode(audio);
    std::printf("Actual:    %zu tokens\n", actual.size());

    if (actual.size() != static_cast<size_t>(T_tok)) {
        std::printf("FAIL: token count mismatch (%zu vs %d)\n",
                    actual.size(), T_tok);
        return 1;
    }

    std::printf("Expected first 10: ");
    for (int i = 0; i < std::min(10, T_tok); ++i)
        std::printf("%d ", expected[i]);
    std::printf("\nActual first 10:   ");
    for (int i = 0; i < std::min(10, T_tok); ++i)
        std::printf("%d ", actual[i]);
    std::printf("\n");

    int mismatch = 0;
    for (int i = 0; i < T_tok; ++i) {
        if (actual[i] != expected[i]) ++mismatch;
    }
    std::printf("\n%d / %d tokens match (%d mismatches)\n",
                T_tok - mismatch, T_tok, mismatch);

    // Also sanity-check range.
    for (int i = 0; i < T_tok; ++i) {
        if (actual[i] < 0 || actual[i] >= chatterbox::S3Tokenizer::VOCAB_SIZE) {
            std::printf("FAIL: actual[%d] = %d is out of range [0, %d)\n",
                        i, actual[i], chatterbox::S3Tokenizer::VOCAB_SIZE);
            return 1;
        }
    }

    if (mismatch > 0) {
        std::printf("\nFAILED: token id sequences differ.\n");
        // Show all mismatches for diagnosis.
        for (int i = 0; i < T_tok; ++i) {
            if (actual[i] != expected[i]) {
                std::printf("  idx %d:  expected=%d  actual=%d  (diff=%d)\n",
                            i, expected[i], actual[i],
                            actual[i] - expected[i]);
            }
        }
        return 1;
    }
    std::printf("\nPASS\n");
    return 0;
}
