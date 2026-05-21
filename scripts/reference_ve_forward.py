"""
NumPy reference forward pass for Chatterbox VoiceEncoder.

Pure-NumPy implementation of upstream models/voice_encoder/voice_encoder.py
on a deterministic seeded mel input. Used to validate the C++ port.

Architecture (from upstream config):
    3-layer LSTM (hidden=256), input mel dim 40
    -> last_hidden of last layer (256,)
    -> Linear(256 -> 256)
    -> ReLU (ve_final_relu=True for Turbo)
    -> L2 normalize

Each PyTorch LSTM layer's weight_ih has shape (4*hidden, input_size)
gate-stacked as i/f/g/o (input, forget, cell, output). Same gate order
applies to weight_hh and bias_ih/bias_hh.

Output: tests/ve_reference.bin (binary):
    int32  T                     mel frame count
    int32  n_mels
    float32 * (T * n_mels)       mel data (row-major)
    int32  emb_dim
    float32 * emb_dim            L2-normalized speaker embedding
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

REPO_ID = "ResembleAI/chatterbox-turbo"
VE_FILE = "ve.safetensors"

N_MELS         = 40
HIDDEN         = 256
N_LSTM_LAYERS  = 3
SPEAKER_EMB    = 256


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def lstm_layer(x: np.ndarray,
               W_ih: np.ndarray, W_hh: np.ndarray,
               b_ih: np.ndarray, b_hh: np.ndarray,
               H: int = HIDDEN):
    """Run one LSTM layer over T frames. Returns (h_seq, last_h, last_c)."""
    T = x.shape[0]
    h = np.zeros(H, dtype=np.float32)
    c = np.zeros(H, dtype=np.float32)
    bias = (b_ih + b_hh).astype(np.float32)
    out = np.zeros((T, H), dtype=np.float32)
    for t in range(T):
        gates = W_ih @ x[t] + W_hh @ h + bias  # (4H,)
        i = sigmoid(gates[0 * H : 1 * H])
        f = sigmoid(gates[1 * H : 2 * H])
        g = np.tanh (gates[2 * H : 3 * H])
        o = sigmoid(gates[3 * H : 4 * H])
        c = f * c + i * g
        h = o * np.tanh(c)
        out[t] = h
    return out, h, c


def ve_forward(state: dict[str, np.ndarray], mel: np.ndarray) -> np.ndarray:
    x = mel.astype(np.float32)
    last_h = None
    for layer in range(N_LSTM_LAYERS):
        W_ih = state[f"lstm.weight_ih_l{layer}"]
        W_hh = state[f"lstm.weight_hh_l{layer}"]
        b_ih = state[f"lstm.bias_ih_l{layer}"]
        b_hh = state[f"lstm.bias_hh_l{layer}"]
        x, last_h, _ = lstm_layer(x, W_ih, W_hh, b_ih, b_hh)

    proj_w = state["proj.weight"]   # (256, 256)
    proj_b = state["proj.bias"]     # (256,)
    y = proj_w @ last_h + proj_b
    y = np.maximum(y, 0.0)           # ReLU (Turbo: ve_final_relu = True)

    nrm = float(np.linalg.norm(y))
    if nrm > 0:
        y = y / nrm
    return y.astype(np.float32)


def make_test_mel(T: int = 50, n_mels: int = N_MELS) -> np.ndarray:
    """Seeded random mel input — pure parity-test fodder, not real audio."""
    rng = np.random.RandomState(7)
    return rng.standard_normal((T, n_mels)).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-bin",  default="tests/ve_reference.bin")
    ap.add_argument("--out-meta", default="tests/ve_reference.json")
    ap.add_argument("--frames",   type=int, default=50)
    args = ap.parse_args()

    print(f"Fetching {VE_FILE} ...")
    src = Path(hf_hub_download(REPO_ID, VE_FILE))
    state = load_file(str(src))
    print(f"  loaded {len(state)} tensors")

    mel = make_test_mel(T=args.frames)
    print(f"VE forward on mel: {mel.shape}")

    emb = ve_forward(state, mel)
    print(f"  embedding shape: {emb.shape}  norm: {float(np.linalg.norm(emb)):.6f}")
    print(f"  min/max:         {float(emb.min()):+.4f} / {float(emb.max()):+.4f}")
    top5 = np.argsort(-emb)[:5]
    print(f"  top-5 ids:        {top5.tolist()}")
    print(f"  top-5 vals:       {emb[top5].tolist()}")

    out_bin = Path(args.out_bin)
    out_bin.parent.mkdir(parents=True, exist_ok=True)
    with open(out_bin, "wb") as f:
        f.write(np.int32(mel.shape[0]).tobytes())
        f.write(np.int32(mel.shape[1]).tobytes())
        f.write(mel.tobytes())
        f.write(np.int32(emb.shape[0]).tobytes())
        f.write(emb.tobytes())

    Path(args.out_meta).write_text(json.dumps({
        "mel_frames": int(mel.shape[0]),
        "mel_dim":    int(mel.shape[1]),
        "embedding_dim":   int(emb.shape[0]),
        "embedding_norm":  float(np.linalg.norm(emb)),
        "top5_idx":  top5.tolist(),
        "top5_val":  emb[top5].tolist(),
        "min":       float(emb.min()),
        "max":       float(emb.max()),
    }, indent=2), encoding="utf-8")

    print(f"\nWrote {out_bin} ({out_bin.stat().st_size} bytes)")
    print(f"Wrote {args.out_meta}")


if __name__ == "__main__":
    main()
