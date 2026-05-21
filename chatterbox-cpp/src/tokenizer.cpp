#include "tokenizer.h"

#include "gguf.h"

#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstring>
#include <utility>

namespace chatterbox {

namespace {

// ---- UTF-8 codec (just what we need; no full validator) ----

struct Utf8 { uint32_t cp; size_t len; };

Utf8 utf8_decode(const std::string& s, size_t pos) {
    if (pos >= s.size()) return {0, 0};
    uint8_t b0 = static_cast<uint8_t>(s[pos]);
    if (b0 < 0x80)            return {b0, 1};
    if ((b0 >> 5) == 0x06 && pos + 1 < s.size()) {
        uint32_t cp = static_cast<uint32_t>(b0 & 0x1F) << 6;
        cp |= static_cast<uint8_t>(s[pos + 1]) & 0x3F;
        return {cp, 2};
    }
    if ((b0 >> 4) == 0x0E && pos + 2 < s.size()) {
        uint32_t cp = static_cast<uint32_t>(b0 & 0x0F) << 12;
        cp |= (static_cast<uint8_t>(s[pos + 1]) & 0x3F) << 6;
        cp |= static_cast<uint8_t>(s[pos + 2]) & 0x3F;
        return {cp, 3};
    }
    if ((b0 >> 3) == 0x1E && pos + 3 < s.size()) {
        uint32_t cp = static_cast<uint32_t>(b0 & 0x07) << 18;
        cp |= (static_cast<uint8_t>(s[pos + 1]) & 0x3F) << 12;
        cp |= (static_cast<uint8_t>(s[pos + 2]) & 0x3F) << 6;
        cp |= static_cast<uint8_t>(s[pos + 3]) & 0x3F;
        return {cp, 4};
    }
    // Invalid lead byte — treat as 1-byte to make forward progress.
    return {b0, 1};
}

std::string utf8_encode(uint32_t cp) {
    std::string r;
    if (cp < 0x80) {
        r.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
        r.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        r.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        r.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        r.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        r.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        r.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        r.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        r.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        r.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
    return r;
}

// ---- Codepoint classification ----
//
// GPT-2's pre-tokenizer regex uses \p{L}, \p{N}, \s, and the complement
// of all three (punctuation/symbol). The Chatterbox Turbo model is
// English-trained; the only non-ASCII letters our test corpus exercises
// are Latin-1 Supplement accents. We cover ASCII + Latin-1 + Latin
// Extended-A/B which is "letters" for any plausible Chatterbox input.
// CJK / Cyrillic / Arabic etc. fall into the punct catch-all here,
// which matches what byte-level BPE would do anyway: each non-ASCII
// byte maps through bytes_to_unicode to a codepoint outside \p{L},
// and BPE handles the rest.

enum CharClass : uint8_t { CC_L, CC_N, CC_P, CC_W };

bool is_letter(uint32_t cp) {
    if ((cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z')) return true;
    if (cp >= 0xC0 && cp <= 0xFF && cp != 0xD7 && cp != 0xF7) return true;  // Latin-1 Sup
    if (cp >= 0x100 && cp <= 0x17F) return true;                            // Latin Ext-A
    if (cp >= 0x180 && cp <= 0x24F) return true;                            // Latin Ext-B
    return false;
}

bool is_digit(uint32_t cp) { return cp >= '0' && cp <= '9'; }

bool is_whitespace(uint32_t cp) {
    return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r'
        || cp == 0x0B || cp == 0x0C;
}

CharClass classify(uint32_t cp) {
    if (is_whitespace(cp)) return CC_W;
    if (is_letter(cp))     return CC_L;
    if (is_digit(cp))      return CC_N;
    return CC_P;
}

}  // namespace

// ---- Tokenizer::load ----

std::unique_ptr<Tokenizer> Tokenizer::load(gguf_context* gguf) {
    if (!gguf) return nullptr;
    auto t = std::unique_ptr<Tokenizer>(new Tokenizer());

    // Tokens
    int64_t tokens_key = gguf_find_key(gguf, "tokenizer.ggml.tokens");
    if (tokens_key < 0) {
        std::fprintf(stderr, "tokenizer: missing tokenizer.ggml.tokens key\n");
        return nullptr;
    }
    const int64_t n_tokens = gguf_get_arr_n(gguf, tokens_key);
    t->id_to_token_.reserve(static_cast<size_t>(n_tokens));
    t->token_to_id_.reserve(static_cast<size_t>(n_tokens));
    for (int64_t i = 0; i < n_tokens; ++i) {
        const char* s = gguf_get_arr_str(gguf, tokens_key, i);
        t->id_to_token_.emplace_back(s);
        t->token_to_id_.emplace(t->id_to_token_.back(), static_cast<int32_t>(i));
    }

    // Merges
    int64_t merges_key = gguf_find_key(gguf, "tokenizer.ggml.merges");
    if (merges_key < 0) {
        std::fprintf(stderr, "tokenizer: missing tokenizer.ggml.merges key\n");
        return nullptr;
    }
    const int64_t n_merges = gguf_get_arr_n(gguf, merges_key);
    t->merge_rank_.reserve(static_cast<size_t>(n_merges));
    for (int64_t i = 0; i < n_merges; ++i) {
        std::string m = gguf_get_arr_str(gguf, merges_key, i);
        const size_t sp = m.find(' ');
        if (sp == std::string::npos || sp == 0 || sp + 1 >= m.size()) {
            std::fprintf(stderr, "tokenizer: malformed merge line: %s\n", m.c_str());
            continue;
        }
        std::string key;
        key.reserve(m.size());
        key.append(m, 0, sp);
        key.push_back('\0');
        key.append(m, sp + 1, std::string::npos);
        t->merge_rank_.emplace(std::move(key), static_cast<int32_t>(i));
    }

    // Special token ids
    auto read_u32 = [&](const char* k, int32_t def) {
        int64_t id = gguf_find_key(gguf, k);
        return id < 0 ? def : static_cast<int32_t>(gguf_get_val_u32(gguf, id));
    };
    t->bos_id_ = read_u32("tokenizer.ggml.bos_token_id",     -1);
    t->eos_id_ = read_u32("tokenizer.ggml.eos_token_id",     -1);
    t->unk_id_ = read_u32("tokenizer.ggml.unknown_token_id", -1);
    t->pad_id_ = read_u32("tokenizer.ggml.padding_token_id", -1);

    // Added-token block (chatterbox_t3-specific). Bracketed paralinguistic
    // tags that must be matched atomically before BPE; longest first.
    int32_t added_first = read_u32("chatterbox_t3.added_tokens.first_id", -1);
    int32_t added_last  = read_u32("chatterbox_t3.added_tokens.last_id",  -1);
    if (added_first >= 0 && added_last >= added_first) {
        for (int32_t i = added_first; i <= added_last
             && static_cast<size_t>(i) < t->id_to_token_.size(); ++i) {
            t->added_by_len_desc_.emplace_back(t->id_to_token_[i], i);
        }
        std::sort(t->added_by_len_desc_.begin(), t->added_by_len_desc_.end(),
                  [](const auto& a, const auto& b) {
                      return a.first.size() > b.first.size();
                  });
    }

    // Bytes-to-unicode table. Reference: GPT-2 encoder.bytes_to_unicode().
    //   Bytes ! .. ~  /  ¡ .. ¬  /  ® .. ÿ      -> same codepoint
    //   Every other byte (control chars, space, etc.) -> sequentially
    //   assigned codepoints in [256, 256+N), where N = count of unmapped.
    bool pretty[256] = {false};
    for (int b = '!';  b <= '~';  ++b) pretty[b] = true;
    for (int b = 0xA1; b <= 0xAC; ++b) pretty[b] = true;
    for (int b = 0xAE; b <= 0xFF; ++b) pretty[b] = true;
    uint32_t spill = 256;
    for (int b = 0; b < 256; ++b) {
        const uint32_t cp = pretty[b] ? static_cast<uint32_t>(b) : spill++;
        t->byte_to_uni_[b] = utf8_encode(cp);
        t->uni_to_byte_[cp] = static_cast<uint8_t>(b);
    }

    return t;
}

// ---- Added-token match ----

std::pair<int32_t, size_t>
Tokenizer::match_added_token(const std::string& text, size_t pos) const {
    for (const auto& [tok, id] : added_by_len_desc_) {
        if (pos + tok.size() <= text.size()
            && std::memcmp(text.data() + pos, tok.data(), tok.size()) == 0) {
            return {id, tok.size()};
        }
    }
    return {-1, 0};
}

// ---- Pre-tokenizer ----
//
// Equivalent to applying GPT-2's
//   '''s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+'''
// regex via re.findall over the input. Hand-written so we don't drag in
// an external regex library — std::regex doesn't reliably support
// \p{L}/\p{N} across implementations.

std::vector<std::string> Tokenizer::pre_tokenize(const std::string& text) const {
    static constexpr const char* CONTRACTIONS[] = {
        "'s", "'t", "'re", "'ve", "'m", "'ll", "'d",
    };

    std::vector<std::string> out;
    const size_t n = text.size();
    size_t pos = 0;

    while (pos < n) {
        // 1. Contractions — tried first to match the regex's left-to-right
        //    alternation. They consume an apostrophe at any position.
        if (text[pos] == '\'') {
            bool matched = false;
            for (const char* c : CONTRACTIONS) {
                const size_t len = std::strlen(c);
                if (pos + len <= n
                    && std::memcmp(text.data() + pos, c, len) == 0) {
                    out.emplace_back(c, len);
                    pos += len;
                    matched = true;
                    break;
                }
            }
            if (matched) continue;
        }

        // 2. ' ?(\p{L}+|\p{N}+|[^\s\p{L}\p{N}]+)'  — same-class run with an
        //    optional leading ASCII space. We decide the class from the
        //    first non-space codepoint and then sweep until a different
        //    class.
        const bool has_space = (text[pos] == ' ');
        const size_t first_cp_pos = has_space ? pos + 1 : pos;

        if (first_cp_pos < n) {
            const Utf8 u = utf8_decode(text, first_cp_pos);
            const CharClass cls = classify(u.cp);
            if (cls != CC_W) {
                size_t scan = first_cp_pos + u.len;
                while (scan < n) {
                    const Utf8 v = utf8_decode(text, scan);
                    if (classify(v.cp) != cls) break;
                    scan += v.len;
                }
                out.emplace_back(text.substr(pos, scan - pos));
                pos = scan;
                continue;
            }
        }

        // 3. Whitespace run. Two sub-cases per the regex:
        //    '\s+(?!\S)'  — run NOT followed by non-whitespace
        //                   (so: trailing whitespace at end of input)
        //    '\s+'        — fallback for mid-text runs
        //
        //    The mid-text case must leave the last whitespace codepoint
        //    for the next iteration to consume as a ' ?' prefix of the
        //    following letter/digit/punct chunk. Otherwise " hello"
        //    becomes ["  ", "hello"] instead of [" ", " hello"].
        size_t ws_end = pos;
        size_t last_ws_start = pos;
        while (ws_end < n) {
            const Utf8 v = utf8_decode(text, ws_end);
            if (classify(v.cp) != CC_W) break;
            last_ws_start = ws_end;
            ws_end += v.len;
        }
        if (ws_end == n) {
            // Trailing whitespace — emit the whole run as one match.
            out.emplace_back(text.substr(pos, ws_end - pos));
            pos = ws_end;
        } else {
            // Mid-text — emit run minus the final codepoint; leave that
            // codepoint for the next iteration as a ' ?' prefix.
            if (last_ws_start > pos) {
                out.emplace_back(text.substr(pos, last_ws_start - pos));
            }
            pos = last_ws_start;
        }
    }
    return out;
}

// ---- BPE on a pre-tokenized word ----

void Tokenizer::encode_word(const std::string& word,
                            std::vector<int32_t>& out) const {
    if (word.empty()) return;

    // Map each input byte to its GPT-2 unicode-encoded piece.
    std::vector<std::string> pieces;
    pieces.reserve(word.size());
    for (unsigned char b : word) pieces.push_back(byte_to_uni_[b]);

    // Iteratively apply the lowest-rank merge over the whole sequence.
    while (pieces.size() >= 2) {
        int32_t best_rank = INT_MAX;
        size_t  best_idx  = 0;
        bool    found     = false;

        std::string key;
        for (size_t i = 0; i + 1 < pieces.size(); ++i) {
            key.assign(pieces[i]);
            key.push_back('\0');
            key.append(pieces[i + 1]);
            auto it = merge_rank_.find(key);
            if (it != merge_rank_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_idx  = i;
                found     = true;
            }
        }
        if (!found) break;

        // Merge every adjacent occurrence of the chosen pair (not only
        // the position found above — matches the upstream BPE pass).
        const std::string& L = pieces[best_idx];
        const std::string& R = pieces[best_idx + 1];

        std::vector<std::string> next;
        next.reserve(pieces.size());
        for (size_t i = 0; i < pieces.size();) {
            if (i + 1 < pieces.size() && pieces[i] == L && pieces[i + 1] == R) {
                next.emplace_back(pieces[i] + pieces[i + 1]);
                i += 2;
            } else {
                next.push_back(std::move(pieces[i]));
                ++i;
            }
        }
        pieces = std::move(next);
    }

    for (const std::string& p : pieces) {
        auto it = token_to_id_.find(p);
        if (it != token_to_id_.end()) {
            out.push_back(it->second);
        } else {
            // Shouldn't happen if byte_to_uni_ matches the vocab. Emit UNK
            // to surface the bug rather than corrupting silently.
            std::fprintf(stderr, "tokenizer: unknown piece %s\n", p.c_str());
            if (unk_id_ >= 0) out.push_back(unk_id_);
        }
    }
}

// ---- encode ----

std::vector<int32_t> Tokenizer::encode(const std::string& text) const {
    std::vector<int32_t> out;
    const size_t n = text.size();
    size_t pos = 0;
    size_t chunk_start = 0;

    auto flush_to = [&](size_t end) {
        if (end <= chunk_start) return;
        const std::string chunk(text, chunk_start, end - chunk_start);
        const auto words = pre_tokenize(chunk);
        for (const auto& w : words) encode_word(w, out);
    };

    while (pos < n) {
        if (text[pos] == '[') {
            auto [id, len] = match_added_token(text, pos);
            if (id >= 0) {
                flush_to(pos);
                out.push_back(id);
                pos += len;
                chunk_start = pos;
                continue;
            }
        }
        ++pos;
    }
    flush_to(pos);
    return out;
}

// ---- decode ----

std::string Tokenizer::decode(const std::vector<int32_t>& ids) const {
    // Concatenate token strings, then walk codepoints reversing the
    // byte-to-unicode map. Mirrors GPT-2's clean_up_tokenization_spaces
    // = false decode.
    std::string concat;
    for (int32_t id : ids) {
        if (id >= 0 && static_cast<size_t>(id) < id_to_token_.size()) {
            concat += id_to_token_[id];
        }
    }
    std::string out;
    out.reserve(concat.size());
    size_t i = 0;
    while (i < concat.size()) {
        const Utf8 u = utf8_decode(concat, i);
        auto it = uni_to_byte_.find(u.cp);
        if (it != uni_to_byte_.end()) {
            out.push_back(static_cast<char>(it->second));
        } else {
            // Codepoint outside the byte-map. Re-emit as UTF-8 verbatim
            // so callers can see the raw token string.
            out.append(concat, i, u.len);
        }
        i += u.len;
    }
    return out;
}

}  // namespace chatterbox
