#pragma once

// GPT-2 byte-level BPE tokenizer for Chatterbox Turbo.
//
// The Turbo tokenizer is stock GPT-2 BPE (50257) plus 19 paralinguistic
// added tokens at IDs 50257..50275 plus <|endoftext|> at 50256. We load
// everything from the standard tokenizer.ggml.* keys baked into the T3
// GGUF by scripts/convert_chatterbox_to_gguf.py, plus the
// chatterbox_t3.added_tokens.{first_id, last_id} range so we know where
// the bracketed atomic-match tokens live.
//
// See docs/tokenizer-findings.md and tests/tokenizer_groundtruth.json.

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

struct gguf_context;

namespace chatterbox {

class Tokenizer {
public:
    // Load all tokenizer state from a chatterbox_t3 GGUF context.
    // Returns nullptr on failure (logs to stderr).
    static std::unique_ptr<Tokenizer> load(gguf_context* gguf);

    // Encode UTF-8 text to a sequence of token ids matching upstream
    // GPT2Tokenizer.encode(text, add_special_tokens=True) — which for
    // GPT-2 / Turbo emits NO automatic BOS/EOS because add_bos_token is
    // false in the upstream config.
    std::vector<int32_t> encode(const std::string& text) const;

    // Concatenate the token strings, then reverse the byte->unicode map.
    // Mirrors HF's clean_up_tokenization_spaces=false behaviour, so the
    // result is byte-exact against the input given fully-reversible token
    // ids. Used for debug / spot-check, not the hot path.
    std::string decode(const std::vector<int32_t>& ids) const;

    int32_t bos_id() const { return bos_id_; }
    int32_t eos_id() const { return eos_id_; }
    size_t  vocab_size() const { return id_to_token_.size(); }

private:
    Tokenizer() = default;

    // ---- byte-level encoding ----
    // For each byte 0..255: its UTF-8 representation after GPT-2's
    // byte-to-unicode mapping. Cached at load time.
    std::string byte_to_uni_[256];
    // Reverse map: the leading codepoint of a byte_to_uni_[b] entry
    // back to b. Used for decode().
    std::unordered_map<uint32_t, uint8_t> uni_to_byte_;

    // ---- vocab + merges ----
    std::vector<std::string> id_to_token_;
    std::unordered_map<std::string, int32_t> token_to_id_;
    // Pair (left, right) -> merge rank (lower = applied earlier). Stored as
    // a concatenated "left\x00right" string key for cheap hashing.
    std::unordered_map<std::string, int32_t> merge_rank_;

    // ---- added tokens (the paralinguistic-tag block) ----
    // Listed as the upstream string keyed on its id. We use this to do an
    // atomic-match pre-pass so e.g. "[laugh]" emits 50275 directly without
    // entering BPE. Ordered by descending length so longest-match wins.
    std::vector<std::pair<std::string, int32_t>> added_by_len_desc_;

    int32_t bos_id_  = -1;
    int32_t eos_id_  = -1;
    int32_t unk_id_  = -1;
    int32_t pad_id_  = -1;

    // ---- internal helpers ----
    // Find the longest added token at text[pos..]. Returns its id and
    // byte length, or {-1, 0} if no match.
    std::pair<int32_t, size_t> match_added_token(const std::string& text,
                                                 size_t pos) const;

    // Split a chunk of text (already known to contain no added tokens)
    // into GPT-2 pre-tokenizer "words", each a UTF-8 substring.
    std::vector<std::string> pre_tokenize(const std::string& text) const;

    // Apply byte->unicode then BPE merges to a pre-tokenizer word.
    // Emits one or more token ids into `out`.
    void encode_word(const std::string& word, std::vector<int32_t>& out) const;
};

}  // namespace chatterbox
