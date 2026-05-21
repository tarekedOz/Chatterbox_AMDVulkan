// Tokenizer parity test against tests/tokenizer_groundtruth.json.
//
// Loads the T3 GGUF (which carries the tokenizer.ggml.* keys), constructs
// a Tokenizer, then runs encode() on each ground-truth sentence and
// compares the resulting token id vector against the expected one.
//
// 12 sentences, all must match for the test to pass.
//
// The JSON parser here is a deliberately minimal special-purpose reader
// for our specific schema (top-level array of {"text": str, "token_ids":
// [int...], "decoded_roundtrip": str}). Bringing in a real JSON library
// for one test file is not worth the dependency surface.

#include "model.h"
#include "tokenizer.h"

#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

struct Case {
    std::string text;
    std::vector<int32_t> expected_ids;
};

// ---- tiny JSON parser ----

class JsonReader {
public:
    explicit JsonReader(std::string src) : s_(std::move(src)) {}

    void skip_ws() {
        while (i_ < s_.size() && std::isspace(static_cast<unsigned char>(s_[i_]))) ++i_;
    }

    bool peek(char c) {
        skip_ws();
        return i_ < s_.size() && s_[i_] == c;
    }

    bool eat(char c) {
        if (peek(c)) { ++i_; return true; }
        return false;
    }

    void expect(char c) {
        if (!eat(c)) {
            std::fprintf(stderr, "JSON: expected '%c' at byte %zu\n", c, i_);
            std::exit(2);
        }
    }

    std::string read_string() {
        expect('"');
        std::string out;
        while (i_ < s_.size() && s_[i_] != '"') {
            if (s_[i_] == '\\' && i_ + 1 < s_.size()) {
                const char esc = s_[i_ + 1];
                i_ += 2;
                switch (esc) {
                    case '"': out.push_back('"'); break;
                    case '\\': out.push_back('\\'); break;
                    case '/': out.push_back('/'); break;
                    case 'b': out.push_back('\b'); break;
                    case 'f': out.push_back('\f'); break;
                    case 'n': out.push_back('\n'); break;
                    case 'r': out.push_back('\r'); break;
                    case 't': out.push_back('\t'); break;
                    case 'u': {
                        // \uXXXX surrogate-aware decoder.
                        if (i_ + 4 > s_.size()) {
                            std::fprintf(stderr, "JSON: short \\u escape\n");
                            std::exit(2);
                        }
                        uint32_t cp = static_cast<uint32_t>(std::stoul(
                            s_.substr(i_, 4), nullptr, 16));
                        i_ += 4;
                        // Handle UTF-16 surrogate pair "\\uD83D\\uDC4D"-style
                        if (cp >= 0xD800 && cp <= 0xDBFF
                            && i_ + 6 <= s_.size()
                            && s_[i_] == '\\' && s_[i_ + 1] == 'u') {
                            uint32_t lo = static_cast<uint32_t>(std::stoul(
                                s_.substr(i_ + 2, 4), nullptr, 16));
                            i_ += 6;
                            cp = 0x10000
                                + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                        }
                        // Encode the cp as UTF-8 directly into out.
                        if (cp < 0x80) {
                            out.push_back(static_cast<char>(cp));
                        } else if (cp < 0x800) {
                            out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
                            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
                        } else if (cp < 0x10000) {
                            out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
                            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
                            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
                        } else {
                            out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
                            out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
                            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
                            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
                        }
                        break;
                    }
                    default:
                        std::fprintf(stderr, "JSON: unknown escape \\%c\n", esc);
                        std::exit(2);
                }
            } else {
                out.push_back(s_[i_]);
                ++i_;
            }
        }
        expect('"');
        return out;
    }

    int32_t read_int() {
        skip_ws();
        size_t start = i_;
        if (i_ < s_.size() && s_[i_] == '-') ++i_;
        while (i_ < s_.size() && std::isdigit(static_cast<unsigned char>(s_[i_]))) ++i_;
        return std::stoi(s_.substr(start, i_ - start));
    }

private:
    std::string s_;
    size_t i_ = 0;
};

std::vector<Case> parse_groundtruth(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::fprintf(stderr, "Cannot open %s\n", path.c_str());
        std::exit(2);
    }
    std::stringstream ss;
    ss << in.rdbuf();
    JsonReader r(ss.str());

    std::vector<Case> cases;
    r.expect('[');
    if (!r.peek(']')) {
        for (;;) {
            r.expect('{');
            Case c;
            bool got_text = false, got_ids = false;
            for (;;) {
                std::string key = r.read_string();
                r.expect(':');
                if (key == "text") {
                    c.text = r.read_string();
                    got_text = true;
                } else if (key == "token_ids") {
                    r.expect('[');
                    if (!r.peek(']')) {
                        for (;;) {
                            c.expected_ids.push_back(r.read_int());
                            if (!r.eat(',')) break;
                        }
                    }
                    r.expect(']');
                    got_ids = true;
                } else if (key == "decoded_roundtrip") {
                    (void)r.read_string();  // ignored
                } else {
                    std::fprintf(stderr, "JSON: unexpected key %s\n", key.c_str());
                    std::exit(2);
                }
                if (!r.eat(',')) break;
            }
            r.expect('}');
            if (!got_text || !got_ids) {
                std::fprintf(stderr, "JSON: entry missing required fields\n");
                std::exit(2);
            }
            cases.push_back(std::move(c));
            if (!r.eat(',')) break;
        }
    }
    r.expect(']');
    return cases;
}

std::string ellipsize(const std::string& s, size_t max) {
    if (s.size() <= max) return s;
    return s.substr(0, max - 3) + "...";
}

std::string ids_to_string(const std::vector<int32_t>& v, size_t cap = 32) {
    std::string out;
    const size_t k = std::min(v.size(), cap);
    for (size_t i = 0; i < k; ++i) {
        out += std::to_string(v[i]);
        if (i + 1 < k) out += ' ';
    }
    if (v.size() > cap) out += " ...";
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "Usage: %s <t3.gguf> <tokenizer_groundtruth.json>\n",
                     argv[0]);
        return 2;
    }

    auto m = chatterbox::load_model(argv[1]);
    if (!m) return 1;

    auto tok = chatterbox::Tokenizer::load(m->gguf());
    if (!tok) {
        std::fprintf(stderr, "Failed to construct tokenizer.\n");
        return 1;
    }
    std::printf("Tokenizer loaded: vocab=%zu  bos/eos=%d/%d\n",
                tok->vocab_size(), tok->bos_id(), tok->eos_id());

    auto cases = parse_groundtruth(argv[2]);
    std::printf("Loaded %zu ground-truth cases\n\n", cases.size());

    size_t passed = 0;
    for (size_t i = 0; i < cases.size(); ++i) {
        const auto& c = cases[i];
        auto actual = tok->encode(c.text);
        const bool ok = actual == c.expected_ids;
        std::printf("[%s] case %zu  (%zu->%zu ids)  %s\n",
                    ok ? "PASS" : "FAIL", i,
                    c.expected_ids.size(), actual.size(),
                    ellipsize(c.text, 50).c_str());
        if (!ok) {
            std::printf("        expected: %s\n",
                        ids_to_string(c.expected_ids).c_str());
            std::printf("        actual:   %s\n",
                        ids_to_string(actual).c_str());
        }
        if (ok) ++passed;
    }

    std::printf("\n%zu/%zu cases passed\n", passed, cases.size());
    return passed == cases.size() ? 0 : 1;
}
