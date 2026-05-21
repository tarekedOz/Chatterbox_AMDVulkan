"""
Tokenizer parity ground-truth dump.

Downloads the upstream Chatterbox Turbo tokenizer from HuggingFace, runs
it against a fixed set of test sentences, and writes two artifacts under
tests/ that become the oracle our C++ tokenizer must reproduce exactly.

Outputs:
  - tests/tokenizer_groundtruth.json   sentence -> token_ids list
  - tests/tokenizer_metadata.json      vocab size, special-token table,
                                       and SHA256 of every upstream
                                       tokenizer artifact (vocab.json,
                                       merges.txt, special_tokens_map.json,
                                       added_tokens.json, tokenizer_config.json)

The metadata file pins the exact upstream tokenizer version (per-file SHA),
so parity regressions can distinguish "our tokenizer drifted" from "upstream
shipped new tokenizer files."
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

REPO_ID = "ResembleAI/chatterbox-turbo"
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "tests"

# The Turbo tokenizer ships as a GPT-2-style "slow" tokenizer: separate
# vocab + merges files plus three small JSON configs. Hash each one so
# we can detect upstream churn precisely.
TOKENIZER_FILES = [
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
    "tokenizer_config.json",
]

# Diverse set of sentences chosen to stress likely tokenizer divergence
# points: punctuation handling, casing, digits, smart quotes, accented
# Latin, paralinguistic tags (Turbo-native per upstream README), and
# whitespace edge cases.
SENTENCES: list[str] = [
    # Plain ASCII baseline
    "Hello world.",

    # The design-doc benchmark sentence
    "Hello, this is a test of the Chatterbox text-to-speech system.",

    # Mixed punctuation, contractions, possessives
    "Wait... what?! 'Are you sure?', she asked. It's John's book.",

    # Digits and units
    "The year is 2026 and we have 12 hours left to ship v1.",

    # Casing
    "AMD Strix Halo runs Vulkan well; NO ROCm REQUIRED.",

    # Em dash + curly quotes (typographic Unicode)
    "He said “it’ll be fine” — then walked off.",

    # Accented Latin
    "Café, résumé, naïve façade.",

    # Paralinguistic tags (Turbo-native)
    "[laugh] That was a good one. [sigh] Anyway.",

    # Whitespace edge cases
    "",
    "   ",
    "  hello  ",

    # Long-form (stresses longer sequences; useful when we wire KV cache)
    "The quick brown fox jumps over the lazy dog. " * 10,
]


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(REPO_ID)

    file_hashes: dict[str, str] = {}
    for fname in TOKENIZER_FILES:
        path = Path(hf_hub_download(REPO_ID, fname))
        file_hashes[fname] = sha256_of(path)

    groundtruth = []
    for s in SENTENCES:
        ids = tokenizer.encode(s, add_special_tokens=True)
        groundtruth.append(
            {
                "text": s,
                "token_ids": ids,
                "decoded_roundtrip": tokenizer.decode(ids, skip_special_tokens=False),
            }
        )

    metadata = {
        "source_repo": REPO_ID,
        "tokenizer_file_sha256": file_hashes,
        "tokenizer_class": type(tokenizer).__name__,
        "vocab_size": tokenizer.vocab_size,
        "model_max_length": getattr(tokenizer, "model_max_length", None),
        "special_tokens": {
            "bos": tokenizer.bos_token,
            "eos": tokenizer.eos_token,
            "unk": tokenizer.unk_token,
            "pad": tokenizer.pad_token,
            "sep": tokenizer.sep_token,
            "cls": tokenizer.cls_token,
            "mask": tokenizer.mask_token,
        },
        "all_special_tokens": tokenizer.all_special_tokens,
        "all_special_ids": tokenizer.all_special_ids,
        "encode_kwargs": {"add_special_tokens": True},
    }

    gt_out = OUT_DIR / "tokenizer_groundtruth.json"
    md_out = OUT_DIR / "tokenizer_metadata.json"
    gt_out.write_text(json.dumps(groundtruth, indent=2, ensure_ascii=False), encoding="utf-8")
    md_out.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote {gt_out}")
    print(f"Wrote {md_out}")
    print(f"tokenizer class:        {type(tokenizer).__name__}")
    print(f"vocab size:             {tokenizer.vocab_size}")
    print(f"# sentences:            {len(SENTENCES)}")
    print("file SHA256s:")
    for name, sha in file_hashes.items():
        print(f"  {name:<28} {sha}")


if __name__ == "__main__":
    main()
