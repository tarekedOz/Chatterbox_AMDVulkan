"""
Convert Chatterbox Turbo PyTorch weights into a GGUF file.

Phase 1.B scope: T3 (autoregressive backbone) only.
Phase 1.C (later): VoiceEncoder + S3Gen.

Design choices:

  - Custom GGUF architecture name: "chatterbox_t3". We are NOT trying to
    masquerade as llama.cpp's stock gpt2 arch — Chatterbox layers
    text/speech embeddings + cond_enc + dual heads outside the
    transformer body, and our chatterbox.cpp loader will know its own
    arch.

  - No transpose at conversion time. The HF GPT-2 implementation uses
    Conv1D layers whose weights live in [in, out] order rather than
    nn.Linear's [out, in]. We preserve the upstream byte layout exactly
    and document the convention; chatterbox.cpp handles layout when it
    sets up ggml_mul_mat operations. This keeps the converter trivially
    reversible.

  - tfmr.wte is SKIPPED: upstream's load path does
    `del t3.tfmr.wte` after loading because the model uses the separate
    text_emb / speech_emb tables for input embeddings. Including wte in
    GGUF would just bloat the file and confuse the loader.

  - Default output dtype: fp16. fp32 is available via --dtype f32.
    Block quantizations (Q5_K_M, Q4_K_M) are produced post-hoc by
    llama.cpp's `quantize` tool against the fp16 GGUF.

Usage:
    python scripts/convert_chatterbox_to_gguf.py \
        --out models/chatterbox-turbo-t3-fp16.gguf

Outputs the file plus a one-line summary on stdout.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Windows consoles default to cp1252; the writer prints non-ASCII status
# and some upstream metadata strings. Force UTF-8 so we don't crash mid-run.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from huggingface_hub import hf_hub_download
from safetensors.numpy import load_file

import gguf

REPO_ID = "ResembleAI/chatterbox-turbo"
T3_FILE = "t3_turbo_v1.safetensors"
ARCH = "chatterbox_t3"

# --- Architecture constants (mirrored from upstream tts_turbo.py + the yaml) ---
N_LAYERS = 24            # GPT2_medium block count
N_HEADS = 16             # GPT2_medium head count
EMBED_DIM = 1024         # n_embd
FFN_DIM = 4096           # GPT2's 4x ffn
CONTEXT_LEN = 8196       # tfmr.wpe.weight rows -- extended past GPT2's 1024
TEXT_VOCAB = 50276       # 50257 stock GPT-2 + 19 paralinguistic added tokens
SPEECH_VOCAB = 6563
SPEAKER_EMB_DIM = 256

# Tokenizer metadata. The Turbo tokenizer is plain GPT-2 BPE plus 19
# paralinguistic added tokens (IDs 50257..50275) and one special token
# <|endoftext|> at 50256. See docs/tokenizer-findings.md.
TOKENIZER_FILES = [
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
]
ENDOFTEXT_ID = 50256
ADDED_TOKENS_FIRST_ID = 50257
ADDED_TOKENS_LAST_ID = 50275

# --- State-dict -> GGUF tensor name mapping ---

# Tensors whose upstream key contains no layer index:
NAME_MAP_STATIC: dict[str, str] = {
    "text_emb.weight":          "token_embd.text.weight",
    "speech_emb.weight":        "token_embd.speech.weight",
    "text_head.weight":         "text_head.weight",
    "speech_head.weight":       "speech_head.weight",
    "speech_head.bias":         "speech_head.bias",
    "cond_enc.spkr_enc.weight": "cond_enc.weight",
    "cond_enc.spkr_enc.bias":   "cond_enc.bias",
    "tfmr.wpe.weight":          "position_embd.weight",
    "tfmr.ln_f.weight":         "output_norm.weight",
    "tfmr.ln_f.bias":           "output_norm.bias",
}

# Per-layer patterns (str.format with i=layer_index):
NAME_MAP_LAYER: dict[str, str] = {
    "tfmr.h.{i}.ln_1.weight":         "blk.{i}.attn_norm.weight",
    "tfmr.h.{i}.ln_1.bias":           "blk.{i}.attn_norm.bias",
    "tfmr.h.{i}.attn.c_attn.weight":  "blk.{i}.attn_qkv.weight",
    "tfmr.h.{i}.attn.c_attn.bias":    "blk.{i}.attn_qkv.bias",
    "tfmr.h.{i}.attn.c_proj.weight":  "blk.{i}.attn_output.weight",
    "tfmr.h.{i}.attn.c_proj.bias":    "blk.{i}.attn_output.bias",
    "tfmr.h.{i}.ln_2.weight":         "blk.{i}.ffn_norm.weight",
    "tfmr.h.{i}.ln_2.bias":           "blk.{i}.ffn_norm.bias",
    "tfmr.h.{i}.mlp.c_fc.weight":     "blk.{i}.ffn_up.weight",
    "tfmr.h.{i}.mlp.c_fc.bias":       "blk.{i}.ffn_up.bias",
    "tfmr.h.{i}.mlp.c_proj.weight":   "blk.{i}.ffn_down.weight",
    "tfmr.h.{i}.mlp.c_proj.bias":     "blk.{i}.ffn_down.bias",
}

# Explicitly dropped — upstream deletes wte after load_state_dict:
SKIP: set[str] = {"tfmr.wte.weight"}


def build_name_map() -> dict[str, str]:
    m = dict(NAME_MAP_STATIC)
    for i in range(N_LAYERS):
        for src_pat, dst_pat in NAME_MAP_LAYER.items():
            m[src_pat.format(i=i)] = dst_pat.format(i=i)
    return m


def load_tokenizer_assets() -> dict[str, object]:
    """Download GPT-2 BPE vocab + merges + added tokens, build a single
    50276-entry tokens list and the BPE merges list. Used to embed
    tokenizer.ggml.* keys into the GGUF so chatterbox.cpp can load
    everything text-side from one file."""
    paths: dict[str, Path] = {}
    for f in TOKENIZER_FILES:
        paths[f] = Path(hf_hub_download(REPO_ID, f))

    vocab: dict[str, int] = json.loads(paths["vocab.json"].read_text(encoding="utf-8"))
    added: dict[str, int] = json.loads(paths["added_tokens.json"].read_text(encoding="utf-8"))

    # Build [id -> token] for the 50276-entry merged vocab.
    # IDs 0..50255 from vocab.json, 50256 is <|endoftext|>, 50257..50275 are
    # the paralinguistic tags from added_tokens.json.
    by_id: dict[int, str] = {}
    for token, tid in vocab.items():
        by_id[tid] = token
    by_id[ENDOFTEXT_ID] = "<|endoftext|>"
    for token, tid in added.items():
        if tid < ADDED_TOKENS_FIRST_ID or tid > ADDED_TOKENS_LAST_ID:
            raise ValueError(f"Added token {token!r} has unexpected id {tid}")
        by_id[tid] = token

    expected_ids = set(range(TEXT_VOCAB))
    actual_ids = set(by_id)
    if actual_ids != expected_ids:
        missing = expected_ids - actual_ids
        extra = actual_ids - expected_ids
        raise ValueError(
            f"Token-id coverage off: missing={sorted(missing)[:5]}... "
            f"extra={sorted(extra)[:5]}..."
        )
    tokens = [by_id[i] for i in range(TEXT_VOCAB)]

    # Parse merges.txt. Standard GPT-2 format: header comment line, then
    # one space-separated pair per line. We pass through verbatim (the
    # llama.cpp / ggml BPE loader expects this exact format).
    merges_raw = paths["merges.txt"].read_text(encoding="utf-8").splitlines()
    merges = [line for line in merges_raw if line and not line.startswith("#")]

    return {
        "tokens": tokens,
        "merges": merges,
        "n_vocab": len(tokens),
        "n_merges": len(merges),
    }


def add_tokenizer_to_writer(writer: gguf.GGUFWriter, tok: dict) -> None:
    """Emit standard tokenizer.ggml.* keys plus chatterbox-specific
    added-tokens range metadata."""
    writer.add_tokenizer_model("gpt2")
    writer.add_tokenizer_pre("gpt-2")
    writer.add_token_list(tok["tokens"])
    writer.add_token_merges(tok["merges"])

    # GPT-2 reuses <|endoftext|> (id 50256) for every special slot.
    writer.add_bos_token_id(ENDOFTEXT_ID)
    writer.add_eos_token_id(ENDOFTEXT_ID)
    writer.add_unk_token_id(ENDOFTEXT_ID)
    writer.add_pad_token_id(ENDOFTEXT_ID)
    writer.add_add_bos_token(False)
    writer.add_add_space_prefix(False)

    # Paralinguistic added tokens: contiguous block [50257..50275]. The
    # C++ loader uses this to short-circuit BPE — text matching one of
    # these tokens emits the id directly.
    writer.add_uint32(f"{ARCH}.added_tokens.first_id", ADDED_TOKENS_FIRST_ID)
    writer.add_uint32(f"{ARCH}.added_tokens.last_id", ADDED_TOKENS_LAST_ID)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="models/chatterbox-turbo-t3-fp16.gguf",
                    help="Output GGUF path (default: %(default)s)")
    ap.add_argument("--dtype", choices=("f16", "f32"), default="f16",
                    help="Tensor dtype in output (default: f16)")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Fetching {T3_FILE} from {REPO_ID} ...")
    src = Path(hf_hub_download(REPO_ID, T3_FILE))
    print(f"Loading {src.name}")
    state = load_file(str(src))
    print(f"  {len(state)} tensors loaded")

    name_map = build_name_map()

    unmapped = sorted(k for k in state if k not in name_map and k not in SKIP)
    if unmapped:
        print("ERROR: upstream tensors with no mapping:")
        for k in unmapped:
            print(f"  {k}  shape={state[k].shape} dtype={state[k].dtype}")
        sys.exit(1)

    missing = sorted(k for k in name_map if k not in state)
    if missing:
        print("ERROR: expected upstream tensors missing from checkpoint:")
        for k in missing:
            print(f"  {k}")
        sys.exit(1)

    target_dtype = np.float16 if args.dtype == "f16" else np.float32
    file_type = (
        gguf.LlamaFileType.MOSTLY_F16
        if args.dtype == "f16"
        else gguf.LlamaFileType.ALL_F32
    )

    writer = gguf.GGUFWriter(str(out_path), arch=ARCH)

    # --- General metadata ---
    writer.add_name("chatterbox-t3-turbo")
    writer.add_description(
        "Chatterbox Turbo T3 (autoregressive backbone). "
        "Converted from " + REPO_ID + "."
    )
    writer.add_source_repo_url("https://github.com/resemble-ai/chatterbox")
    writer.add_file_type(file_type)

    # --- Architecture metadata ---
    # add_block_count etc. prepend the arch name automatically, producing
    # keys like 'chatterbox_t3.block_count'.
    writer.add_block_count(N_LAYERS)
    writer.add_context_length(CONTEXT_LEN)
    writer.add_embedding_length(EMBED_DIM)
    writer.add_feed_forward_length(FFN_DIM)
    writer.add_head_count(N_HEADS)
    writer.add_layer_norm_eps(1e-5)  # GPT-2 default
    writer.add_vocab_size(TEXT_VOCAB)

    # Chatterbox-specific keys (no built-in helper) — prefix manually.
    writer.add_uint32(f"{ARCH}.speech_vocab_size", SPEECH_VOCAB)
    writer.add_uint32(f"{ARCH}.speaker_emb_dim", SPEAKER_EMB_DIM)
    writer.add_string(f"{ARCH}.tensor_layout", "upstream_native")

    # --- Tokenizer metadata ---
    # Embedded into T3 because T3 is the consumer of text tokens. The
    # other component GGUFs (ve, s3gen) deliberately omit tokenizer keys.
    print("Loading tokenizer assets from HF ...")
    tok = load_tokenizer_assets()
    print(f"  {tok['n_vocab']} tokens, {tok['n_merges']} merges")
    add_tokenizer_to_writer(writer, tok)

    # --- Tensors ---
    # Convention (matches llama.cpp): 2D matmul weights cast to target dtype
    # (fp16 default); 1D vectors — biases and LayerNorm gains — stay fp32.
    # ggml CPU and Vulkan backends don't accept mixed-dtype ggml_add /
    # ggml_mul (the small ops applied with these vectors), and these
    # tensors are tiny anyway (~few hundred KiB total).
    n_written = 0
    n_bytes = 0
    for src_name, dst_name in sorted(name_map.items()):
        t = state[src_name]
        if t.ndim == 1:
            t = t.astype(np.float32)
        else:
            t = t.astype(target_dtype)
        writer.add_tensor(dst_name, t)
        n_written += 1
        n_bytes += t.nbytes

    print(f"  {n_written} tensors mapped -> {n_bytes / (1024 ** 2):.1f} MiB at {args.dtype}")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(f"\nWrote {out_path} ({out_path.stat().st_size / (1024 ** 2):.1f} MiB)")


if __name__ == "__main__":
    main()
