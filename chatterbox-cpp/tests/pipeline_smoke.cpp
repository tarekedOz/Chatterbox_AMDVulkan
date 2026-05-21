// End-to-end smoke test for the conditioning + generation chain.
//
// Wires together everything that's been ported to C++ so far:
//
//   mel (40, T)   ->  VE  ->  speaker_emb (256, L2-normed)
//                                                    \
//   WAV (16 kHz)  ->  S3Tokenizer  ->  prompt_tokens   } -> T3::prefill
//                                                    /
//   "Hello world."  ->  Tokenizer  ->  text_tokens   /
//
//   T3::prefill(...)  ->  T3::generate(seed=42, max=20)
//     ->  vector<int32_t> speech_token_ids
//
// Not a parity test — no Python oracle for the full pipeline. Asserts
// the loop terminates with valid speech token ids in [0, SPEECH_VOCAB).
// This is the integration check that proves the API surface fits
// together; the next pieces (S3Gen flow/decoder + HiFiGAN vocoder)
// will consume these speech_token_ids to produce actual audio.

#include "model.h"
#include "s3tok.h"
#include "sampler.h"
#include "t3.h"
#include "tokenizer.h"
#include "ve.h"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

namespace {

bool read_ve_reference(const std::string& path,
                        std::vector<float>& mel,
                        int& n_frames, int& n_mels) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    int32_t f = 0, m = 0;
    in.read(reinterpret_cast<char*>(&f), sizeof(f));
    in.read(reinterpret_cast<char*>(&m), sizeof(m));
    mel.assign(static_cast<size_t>(f) * m, 0.0f);
    in.read(reinterpret_cast<char*>(mel.data()), mel.size() * sizeof(float));
    if (!in) return false;
    n_frames = f;
    n_mels   = m;
    return true;
}

bool read_s3_tokenizer_wav(const std::string& path,
                            std::vector<float>& audio) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    int32_t n = 0;
    in.read(reinterpret_cast<char*>(&n), sizeof(n));
    audio.assign(n, 0.0f);
    in.read(reinterpret_cast<char*>(audio.data()), n * sizeof(float));
    return in.good();
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 6) {
        std::fprintf(stderr,
                     "Usage: %s <t3.gguf> <ve.gguf> <s3gen.gguf>\n"
                     "          <ve_reference.bin> <s3_tokenizer_reference.bin>\n",
                     argv[0]);
        return 2;
    }
    const char* t3_path   = argv[1];
    const char* ve_path   = argv[2];
    const char* s3gen_path = argv[3];
    const char* ve_ref     = argv[4];
    const char* s3_ref     = argv[5];

    // ----- Load all three models. -----
    std::printf("Loading models ...\n");
    auto t3_model    = chatterbox::load_model(t3_path);
    auto ve_model    = chatterbox::load_model(ve_path);
    auto s3gen_model = chatterbox::load_model(s3gen_path);
    if (!t3_model || !ve_model || !s3gen_model) return 1;

    auto t3       = chatterbox::T3::load(t3_model.get(), /*max_context=*/2048);
    auto ve       = chatterbox::VE::load(ve_model.get());
    auto s3_tok   = chatterbox::S3Tokenizer::load(s3gen_model.get());
    auto text_tok = chatterbox::Tokenizer::load(t3_model->gguf());
    if (!t3 || !ve || !s3_tok || !text_tok) {
        std::fprintf(stderr, "Component load failed.\n");
        return 1;
    }
    std::printf("  T3:           ok  (n_layer=%d, context=%d)\n",
                t3->config().n_layers, t3->max_context());
    std::printf("  VE:           ok  (n_mels=%d, emb_dim=%d)\n",
                ve->config().n_mels, ve->config().emb_dim);
    std::printf("  S3Tokenizer:  ok  (vocab=%d)\n",
                chatterbox::S3Tokenizer::VOCAB_SIZE);
    std::printf("  Tokenizer:    ok  (vocab=%zu)\n", text_tok->vocab_size());

    // ----- Speaker embedding: mel -> VE. -----
    std::vector<float> ve_mel;
    int ve_frames = 0, ve_n_mels = 0;
    if (!read_ve_reference(ve_ref, ve_mel, ve_frames, ve_n_mels)) {
        std::fprintf(stderr, "Cannot read VE reference: %s\n", ve_ref);
        return 1;
    }
    auto speaker_emb = ve->forward(ve_mel, ve_frames);
    if (speaker_emb.empty()) {
        std::fprintf(stderr, "VE::forward returned empty\n");
        return 1;
    }
    std::printf("\nspeaker_emb: %zu-d (from %d-frame seeded mel)\n",
                speaker_emb.size(), ve_frames);

    // ----- Prompt speech tokens: WAV -> S3Tokenizer. -----
    std::vector<float> wav;
    if (!read_s3_tokenizer_wav(s3_ref, wav)) {
        std::fprintf(stderr, "Cannot read S3 tokenizer reference: %s\n", s3_ref);
        return 1;
    }
    auto prompt_ids = s3_tok->encode(wav);
    if (prompt_ids.empty()) {
        std::fprintf(stderr, "S3Tokenizer::encode returned empty\n");
        return 1;
    }
    std::printf("prompt_speech_tokens: %zu  (first 10: ", prompt_ids.size());
    for (size_t i = 0; i < std::min<size_t>(10, prompt_ids.size()); ++i) {
        std::printf("%d ", prompt_ids[i]);
    }
    std::printf("...)\n");

    // ----- Text tokens: tokenizer encode. -----
    const std::string text = "Hello world.";
    auto text_ids = text_tok->encode(text);
    std::printf("text_tokens for %s: ", ("\"" + text + "\"").c_str());
    for (int32_t t : text_ids) std::printf("%d ", t);
    std::printf("\n");

    // ----- T3 prefill + generate. -----
    auto pre_logits = t3->prefill(speaker_emb, prompt_ids, text_ids);
    if (pre_logits.empty()) {
        std::fprintf(stderr, "T3::prefill returned empty\n");
        return 1;
    }
    std::printf("\nT3 prefill: n_past = %d\n", t3->n_past());

    chatterbox::T3::GenParams gp;
    gp.sampling.seed               = 42;
    gp.sampling.temperature        = 0.8f;
    gp.sampling.top_k              = 50;
    gp.sampling.top_p              = 0.95f;
    gp.sampling.repetition_penalty = 1.2f;
    gp.max_tokens   = 20;
    gp.start_token  = 6561;
    gp.vocab_limit  = 6561;

    auto speech = t3->generate(gp);
    std::printf("Generated %zu speech tokens (n_past=%d): ",
                speech.size(), t3->n_past());
    for (int32_t t : speech) std::printf("%d ", t);
    std::printf("\n");

    // ----- Validity checks. -----
    int failures = 0;
    for (int32_t t : speech) {
        if (t < 0 || t >= gp.vocab_limit) {
            std::printf("FAIL: speech token %d out of range\n", t);
            ++failures;
        }
    }
    if (speech.size() > static_cast<size_t>(gp.max_tokens)) {
        std::printf("FAIL: generated %zu > max_tokens=%d\n",
                    speech.size(), gp.max_tokens);
        ++failures;
    }
    if (t3->n_past() > t3->max_context()) {
        std::printf("FAIL: n_past=%d > max_context=%d\n",
                    t3->n_past(), t3->max_context());
        ++failures;
    }

    if (failures) {
        std::printf("\n%d smoke check(s) failed.\n", failures);
        return 1;
    }
    std::printf("\nPipeline smoke PASS — WAV->mel->S3Tok->prompt_ids /\n");
    std::printf("                       mel->VE->speaker_emb         } -> T3 -> %zu speech ids\n",
                speech.size());
    std::printf("                       text->Tokenizer->text_ids    /\n");
    return 0;
}
