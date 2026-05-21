"""
NumPy reference for the full S3 audio tokenizer pipeline.

Composes the existing mel-extraction and encoder reference scripts
with the FSQ quantizer (FSQVectorQuantization in upstream model_v2.py):

    audio(16 kHz)
      -> log-mel (128, T_mel)              [reference_s3_mel]
      -> AudioEncoderV2 hidden (T_tok, 1280)   [reference_s3_encoder]
      -> FSQ quantizer (per-token):
           h = hidden_t @ proj_down.T + b          # Linear(1280 -> 8)
           h = tanh(h) * 0.9990000128746033       # squash + slight contract
           h = round(h) + 1                       # {-1, 0, 1} -> {0, 1, 2}
           token = sum h[i] * 3^i for i in 0..7    # base-3 encode
      -> token_ids (T_tok,) in [0, 3^8) = [0, 6561)

The 0.999 contraction nudges tanh outputs away from the ±1 boundary so
rounding to {-1, +1} is robust; the +1 offset shifts the integer values
to {0, 1, 2} so the base-3 sum maps cleanly to [0, 6561).

Output: tests/s3_tokenizer_reference.bin
    int32   n_samples
    float32 * n_samples           waveform (16 kHz mono)
    int32   T_tok
    int32 * T_tok                 expected speech token ids
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from huggingface_hub import hf_hub_download
from safetensors.numpy import load_file

sys.path.insert(0, str(Path(__file__).parent))
from reference_s3_mel import log_mel_spectrogram, make_test_wav, S3_SR
from reference_s3_encoder import encoder_forward

REPO_ID = "ResembleAI/chatterbox-turbo"
S3GEN_FILE = "s3gen_meanflow.safetensors"

FSQ_DIM    = 8
FSQ_LEVEL  = 3
FSQ_SCALE  = 0.9990000128746033
FSQ_POWERS = np.array([FSQ_LEVEL ** i for i in range(FSQ_DIM)], dtype=np.int64)
SPEECH_VOCAB_SIZE = FSQ_LEVEL ** FSQ_DIM   # 6561


def fsq_encode(hidden: np.ndarray,
               proj_w: np.ndarray,
               proj_b: np.ndarray) -> np.ndarray:
    """hidden: (T_tok, 1280). Returns (T_tok,) int32 token ids in [0, 6561)."""
    h = hidden @ proj_w.T + proj_b                # (T_tok, 8)
    h = np.tanh(h) * FSQ_SCALE
    h = np.round(h).astype(np.int64) + 1           # {0, 1, 2}
    # Defensive clamp in case rounding produces ±2 at extreme tanh values.
    h = np.clip(h, 0, 2)
    tokens = (h * FSQ_POWERS[None, :]).sum(axis=-1).astype(np.int32)
    return tokens


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-bin",  default="tests/s3_tokenizer_reference.bin")
    ap.add_argument("--out-meta", default="tests/s3_tokenizer_reference.json")
    ap.add_argument("--seconds", type=float, default=1.0)
    args = ap.parse_args()

    print(f"Fetching {S3GEN_FILE} ...")
    src = Path(hf_hub_download(REPO_ID, S3GEN_FILE))
    state_full = load_file(str(src))

    # Pull everything tokenizer-related (mel filters, window, encoder,
    # quantizer) and rename to drop the tokenizer. prefix.
    state = {}
    for k, v in state_full.items():
        if k.startswith("tokenizer."):
            state[k[len("tokenizer."):]] = v.astype(np.float32)
    print(f"  loaded {len(state)} tokenizer tensors")

    mel_filters = state["_mel_filters"]
    window      = state["window"]
    proj_w = state["quantizer._codebook.project_down.weight"]   # (8, 1280)
    proj_b = state["quantizer._codebook.project_down.bias"]      # (8,)

    wav = make_test_wav(seconds=args.seconds)
    log_mel = log_mel_spectrogram(wav, mel_filters, window)
    hidden  = encoder_forward(state, log_mel)
    tokens  = fsq_encode(hidden, proj_w, proj_b)

    print(f"wav:      {wav.shape[0]} samples = {wav.shape[0]/S3_SR:.2f} s")
    print(f"hidden:   {hidden.shape}")
    print(f"tokens:   {tokens.shape}  range [{int(tokens.min())}..{int(tokens.max())}]")
    print(f"  first 10: {tokens[:10].tolist()}")
    print(f"  last 5:   {tokens[-5:].tolist()}")
    assert (tokens >= 0).all() and (tokens < SPEECH_VOCAB_SIZE).all()

    out_bin = Path(args.out_bin)
    out_bin.parent.mkdir(parents=True, exist_ok=True)
    with open(out_bin, "wb") as f:
        f.write(np.int32(wav.shape[0]).tobytes())
        f.write(wav.tobytes())
        f.write(np.int32(tokens.shape[0]).tobytes())
        f.write(tokens.tobytes())

    Path(args.out_meta).write_text(json.dumps({
        "seconds":  float(args.seconds),
        "sample_rate": S3_SR,
        "n_samples": int(wav.shape[0]),
        "T_tok": int(tokens.shape[0]),
        "speech_vocab_size": SPEECH_VOCAB_SIZE,
        "tokens": tokens.tolist(),
        "token_min": int(tokens.min()),
        "token_max": int(tokens.max()),
    }, indent=2), encoding="utf-8")

    print(f"\nWrote {out_bin} ({out_bin.stat().st_size} bytes)")
    print(f"Wrote {args.out_meta}")


if __name__ == "__main__":
    main()
