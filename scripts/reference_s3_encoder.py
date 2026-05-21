"""
NumPy reference for the S3 audio tokenizer's AudioEncoderV2 forward pass.

Phase 1-of-2 of the encoder port (next: FSQ quantizer + glue).
Mirrors xingchensong/S3Tokenizer's model_v2.AudioEncoderV2 exactly:

    log_mel (128, T_mel)
      -> Conv1d(128 -> 1280, k=3, stride=2, pad=1) -> GELU(exact, erf-based)
      -> Conv1d(1280 -> 1280, k=3, stride=2, pad=1) -> GELU
      -> permute (D, T') -> (T', D)
      -> 6 x ResidualAttentionBlock:
           x = x + attn(LN(x), RoPE)[0]
           x = x + mlp(LN(x))
    -> hidden (T_tok, 1280)   where T_tok = T_mel / 4

Each attention block:
  - Q, K, V = Linear projections of LN(x); key has no bias.
  - Q, K reshaped to (T, n_head, head_dim) and rotary-position-embedded
    in the half-split (NEOX) style.
  - FSMN memory branch: depthwise Conv1d(D, D, k=31, groups=D) on V,
    plus V residual, plus mask. Added to the attention output.
  - Scaled (head_dim**-0.25 on BOTH Q and K) Q@K^T -> softmax -> @V.
  - Output linear projection.

For testing we feed the seeded 1-sec synthetic WAV from
scripts/reference_s3_mel.py — same input both sides so we can validate
the C++ implementation end-to-end against this oracle.

Output: tests/s3_encoder_reference.bin
    int32   T_mel
    int32   n_mels
    float32 * (n_mels * T_mel)         log-mel (n_mels, T_mel) row-major
    int32   T_tok
    int32   n_state
    float32 * (T_tok * n_state)        hidden  (T_tok, n_state) row-major
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from huggingface_hub import hf_hub_download
from safetensors.numpy import load_file
from scipy.special import erf

# Reuse the mel module
sys.path.insert(0, str(Path(__file__).parent))
from reference_s3_mel import (
    S3_SR, N_FFT, N_HOP, N_MELS, log_mel_spectrogram, make_test_wav,
)

REPO_ID = "ResembleAI/chatterbox-turbo"
S3GEN_FILE = "s3gen_meanflow.safetensors"

N_STATE   = 1280
N_HEAD    = 20
N_LAYER   = 6
HEAD_DIM  = N_STATE // N_HEAD   # 64
FSMN_K    = 31
ROPE_BASE = 10000.0
ROPE_MAX  = 1024 * 2            # precompute_freqs_cis end=1024*2
LN_EPS    = 1e-5


def gelu_exact(x: np.ndarray) -> np.ndarray:
    """torch.nn.GELU() default — exact erf-based."""
    return 0.5 * x * (1.0 + erf(x / math.sqrt(2.0)))


def layer_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray,
               eps: float = LN_EPS) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    var  = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * weight + bias


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_max = x.max(axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / e.sum(axis=axis, keepdims=True)


def conv1d_s2_p1(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Conv1d(in, out, k=3, stride=2, padding=1) over (in, T) -> (out, T_out).
    Matrix-multiply form (unfold + GEMM), no scipy."""
    out_ch, in_ch, K = weight.shape
    assert K == 3
    T = x.shape[1]
    xp = np.pad(x, ((0, 0), (1, 1)))     # symmetric zero-pad by 1
    T_out = (T + 2 - K) // 2 + 1
    # Unfold: for each output t, gather a (in_ch, K) window from input.
    unfolded = np.empty((T_out, in_ch * K), dtype=np.float32)
    for t_out in range(T_out):
        t_in = t_out * 2
        unfolded[t_out] = xp[:, t_in:t_in + K].reshape(-1)
    w_flat = weight.reshape(out_ch, in_ch * K).astype(np.float32)
    out = unfolded @ w_flat.T + bias
    return out.T.astype(np.float32)


def precompute_freqs_cis(dim: int = HEAD_DIM, end: int = ROPE_MAX,
                         theta: float = ROPE_BASE):
    freqs = 1.0 / (theta ** (np.arange(0, dim, 2).astype(np.float32) / dim))
    t = np.arange(end).astype(np.float32)
    angles = np.outer(t, freqs)           # (end, dim/2)
    return np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)


def apply_rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """x: (T, n_head, head_dim). cos, sin: (T, head_dim/2).
    Half-split NEOX form: pair (i, i + half) uses the same angle."""
    D = x.shape[-1]
    half = D // 2
    cos_full = np.concatenate([cos, cos], axis=-1)[:, None, :]   # (T, 1, D)
    sin_full = np.concatenate([sin, sin], axis=-1)[:, None, :]
    x_l, x_r = x[..., :half], x[..., half:]
    x_rot = np.concatenate([-x_r, x_l], axis=-1)
    return x * cos_full + x_rot * sin_full


def fsmn_memory(v: np.ndarray, fsmn_w: np.ndarray) -> np.ndarray:
    """v: (T, n_head, head_dim) -> (T, n_state). Depthwise conv on the
    flattened (T, D) view, kernel 31 with symmetric (15, 15) zero pad,
    plus a residual add of the input."""
    T, H, Hd = v.shape
    D = H * Hd
    v_flat = v.reshape(T, D)                                 # (T, D)
    pad_left  = (FSMN_K - 1) // 2
    pad_right = FSMN_K - 1 - pad_left
    x = v_flat.T                                              # (D, T)
    x = np.pad(x, ((0, 0), (pad_left, pad_right)))            # (D, T + K - 1)
    # fsmn_w shape (D, 1, K). Per-channel 1D conv.
    w = fsmn_w[:, 0, :]                                       # (D, K)
    out = np.zeros((D, T), dtype=np.float32)
    for k in range(FSMN_K):
        out += x[:, k:k + T] * w[:, k:k + 1]
    return out.T + v_flat                                      # (T, D)


def attention_block(x: np.ndarray, w: dict, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """One FSMN-MHA-with-RoPE block including the inner LN + output proj + FSMN add."""
    T, D = x.shape
    x_n = layer_norm(x, w["attn_ln.weight"], w["attn_ln.bias"])

    # QKV projections. Linear stored as (out, in) so y = x @ W.T + b.
    q = x_n @ w["attn.query.weight"].T + w["attn.query.bias"]
    k = x_n @ w["attn.key.weight"].T                              # no bias
    v = x_n @ w["attn.value.weight"].T + w["attn.value.bias"]

    q = q.reshape(T, N_HEAD, HEAD_DIM)
    k = k.reshape(T, N_HEAD, HEAD_DIM)
    v = v.reshape(T, N_HEAD, HEAD_DIM)

    q = apply_rope(q, cos[:T], sin[:T])
    k = apply_rope(k, cos[:T], sin[:T])

    fsm = fsmn_memory(v, w["attn.fsmn_block.weight"])             # (T, D)

    # Scale Q AND K by head_dim**-0.25 (per source). Net scale on QK^T is **-0.5.
    scale = HEAD_DIM ** -0.25
    qs = q.transpose(1, 0, 2) * scale                             # (H, T, Hd)
    ks = k.transpose(1, 0, 2) * scale
    vs = v.transpose(1, 0, 2)

    qk = qs @ ks.transpose(0, 2, 1)                               # (H, T, T)
    w_attn = softmax(qk, axis=-1)
    attn = (w_attn @ vs).transpose(1, 0, 2).reshape(T, D)         # (T, D)

    out = attn @ w["attn.out.weight"].T + w["attn.out.bias"] + fsm
    return x + out


def mlp_block(x: np.ndarray, w: dict) -> np.ndarray:
    x_n = layer_norm(x, w["mlp_ln.weight"], w["mlp_ln.bias"])
    h = x_n @ w["mlp.0.weight"].T + w["mlp.0.bias"]              # (T, 5120)
    h = gelu_exact(h)
    h = h @ w["mlp.2.weight"].T + w["mlp.2.bias"]                # (T, 1280)
    return x + h


def encoder_forward(state: dict[str, np.ndarray], log_mel: np.ndarray) -> np.ndarray:
    """log_mel: (n_mels, T_mel). Returns (T_tok, n_state)."""
    x = log_mel.astype(np.float32)

    x = conv1d_s2_p1(x, state["encoder.conv1.weight"], state["encoder.conv1.bias"])
    x = gelu_exact(x)
    x = conv1d_s2_p1(x, state["encoder.conv2.weight"], state["encoder.conv2.bias"])
    x = gelu_exact(x)
    x = x.T  # (T_tok, n_state)

    cos, sin = precompute_freqs_cis()

    for i in range(N_LAYER):
        prefix = f"encoder.blocks.{i}."
        w = {k: state[prefix + k] for k in [
            "attn_ln.weight", "attn_ln.bias",
            "attn.query.weight", "attn.query.bias",
            "attn.key.weight",
            "attn.value.weight", "attn.value.bias",
            "attn.out.weight", "attn.out.bias",
            "attn.fsmn_block.weight",
            "mlp_ln.weight", "mlp_ln.bias",
            "mlp.0.weight", "mlp.0.bias",
            "mlp.2.weight", "mlp.2.bias",
        ]}
        x = attention_block(x, w, cos, sin)
        x = mlp_block(x, w)

    return x.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-bin",  default="tests/s3_encoder_reference.bin")
    ap.add_argument("--out-meta", default="tests/s3_encoder_reference.json")
    ap.add_argument("--seconds", type=float, default=1.0)
    args = ap.parse_args()

    print(f"Fetching {S3GEN_FILE} ...")
    src = Path(hf_hub_download(REPO_ID, S3GEN_FILE))
    state_full = load_file(str(src))

    # Pull only the tokenizer.encoder.* tensors (rename for the local fns).
    state = {}
    for k, v in state_full.items():
        if k.startswith("tokenizer.encoder."):
            state[k[len("tokenizer."):]] = v.astype(np.float32)
        elif k == "tokenizer._mel_filters" or k == "tokenizer.window":
            state[k[len("tokenizer."):]] = v.astype(np.float32)
    print(f"  loaded {len(state)} encoder-related tensors")

    mel_filters = state["_mel_filters"]
    window      = state["window"]

    wav = make_test_wav(seconds=args.seconds)
    log_mel = log_mel_spectrogram(wav, mel_filters, window)
    print(f"log-mel:  {log_mel.shape}  (n_mels, T_mel)")

    hidden = encoder_forward(state, log_mel)
    print(f"hidden:   {hidden.shape}  (T_tok, n_state)")
    print(f"  min/max/avg: {float(hidden.min()):+.4f} / {float(hidden.max()):+.4f}"
          f" / {float(hidden.mean()):+.4f}")

    # Top-5 of position 0 row (any reasonable signal to spot-check).
    row0 = hidden[0]
    top5 = np.argsort(-row0)[:5]
    print(f"  hidden[0] top-5 dims: {top5.tolist()}")
    print(f"  hidden[0] top-5 vals: {row0[top5].tolist()}")

    out_bin = Path(args.out_bin)
    out_bin.parent.mkdir(parents=True, exist_ok=True)
    T_mel = int(log_mel.shape[1])
    n_mels = int(log_mel.shape[0])
    T_tok, n_state = int(hidden.shape[0]), int(hidden.shape[1])
    with open(out_bin, "wb") as f:
        f.write(np.int32(T_mel).tobytes())
        f.write(np.int32(n_mels).tobytes())
        f.write(log_mel.astype(np.float32).tobytes())
        f.write(np.int32(T_tok).tobytes())
        f.write(np.int32(n_state).tobytes())
        f.write(hidden.astype(np.float32).tobytes())

    Path(args.out_meta).write_text(json.dumps({
        "seconds": args.seconds,
        "T_mel": T_mel, "n_mels": n_mels,
        "T_tok": T_tok, "n_state": n_state,
        "hidden_min": float(hidden.min()),
        "hidden_max": float(hidden.max()),
        "hidden_mean": float(hidden.mean()),
        "row0_top5_dims": top5.tolist(),
        "row0_top5_vals": row0[top5].tolist(),
    }, indent=2), encoding="utf-8")

    print(f"\nWrote {out_bin} ({out_bin.stat().st_size} bytes)")
    print(f"Wrote {args.out_meta}")


if __name__ == "__main__":
    main()
