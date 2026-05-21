"""
NumPy reference for the Flow input projection + UpsampleConformerEncoder
(plus spk_embed_affine_layer and encoder_proj).

Mirrors upstream chatterbox/models/s3gen/{flow.py, transformer/*} exactly.
Inputs to the C++ port that this oracle anchors:

  - input_embedding:        nn.Embedding(6561, 512)
  - spk_embed_affine_layer: nn.Linear(192, 80)
  - encoder.embed:          LinearNoSubsampling (Linear 512->512 + LN(eps=1e-5))
                            + EspnetRelPositionalEncoding (sin/cos table)
  - encoder.pre_lookahead_layer:
        residual( leaky_relu(conv1[k=4, no-pad, right-pad-3])
                  -> conv2[k=3, no-pad, left-pad-2] )
  - encoder.encoders.{0..5}: ConformerEncoderLayer (norm_before, no macaron,
                              no conv_module, rel-pos attn, swish-FF)
  - encoder.up_layer:        Upsample1D (nearest 2x + left-pad 4 + conv[k=5])
  - encoder.up_embed:        LinearNoSubsampling (same shape as embed)
  - encoder.up_encoders.{0..3}: 4x ConformerEncoderLayer
  - encoder.after_norm:      LayerNorm(eps=1e-5)
  - encoder_proj:            nn.Linear(512, 80)

Output: tests/flow_encoder_reference.bin (binary, stage-keyed dump)

    int32   T_in
    int32   d_in       = 512
    int32   d_out      = 80
    int32   n_stages

    int32[T_in]        tokens               (speech-token ids)
    float32[192]       speaker_emb_normed   (already L2-normalized along the
                                              last axis -- the input to
                                              spk_embed_affine_layer)
    float32[80]        spk_affine_out       (linear output, what gets passed
                                              to the CFM decoder as `spks`)

    for each stage (name + (1, T, C) tensor row-major):
        char[32]   stage_name (NUL-padded)
        int32      dim0=1
        int32      dim1=T
        int32      dim2=C
        float32[dim0*dim1*dim2]
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


REPO_ID    = "ResembleAI/chatterbox-turbo"
S3GEN_FILE = "s3gen_meanflow.safetensors"

VOCAB_SIZE  = 6561
D_MODEL     = 512
D_OUT       = 80
SPK_DIM     = 192
N_HEADS     = 8
D_HEAD      = D_MODEL // N_HEADS    # 64
FF_HIDDEN   = 2048
N_BLOCKS    = 6
N_UP_BLOCKS = 4
LN_EPS      = 1e-5
LN_EPS_ATTN = 1e-12       # encoder_layer's norm_mha/norm_ff use eps=1e-12
PRE_LOOKAHEAD_LEN = 3
UP_STRIDE    = 2
MAX_REL_LEN  = 5000        # default of EspnetRelPositionalEncoding


# ----------------------------------------------------------------------------
# Primitives
# ----------------------------------------------------------------------------

def linear(x: np.ndarray, weight: np.ndarray,
            bias: np.ndarray | None) -> np.ndarray:
    """torch.nn.Linear forward. weight shape (out, in), x last-dim is in."""
    y = x @ weight.T
    if bias is not None:
        y = y + bias
    return y


def layer_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray,
                eps: float) -> np.ndarray:
    """LayerNorm over the last axis."""
    mean = x.mean(axis=-1, keepdims=True)
    var  = x.var (axis=-1, keepdims=True)
    y = (x - mean) / np.sqrt(var + eps)
    return y * weight + bias


def conv1d(x: np.ndarray, weight: np.ndarray, bias: np.ndarray | None,
            stride: int = 1, padding: int = 0,
            dilation: int = 1) -> np.ndarray:
    """x: (C_in, T). weight: (C_out, C_in, K). Returns (C_out, T_out)."""
    Cin, T = x.shape
    Cout, _, K = weight.shape
    xp = np.pad(x, ((0, 0), (padding, padding)))
    Tp = xp.shape[1]
    eff_K = (K - 1) * dilation + 1
    T_out = (Tp - eff_K) // stride + 1
    col = np.empty((Cin * K, T_out), dtype=np.float32)
    for k in range(K):
        start = k * dilation
        end = start + stride * T_out
        col[k * Cin:(k + 1) * Cin, :] = xp[:, start:end:stride]
    w_flat = weight.transpose(2, 1, 0).reshape(K * Cin, Cout)
    out = w_flat.T @ col
    out = out.reshape(Cout, T_out).astype(np.float32)
    if bias is not None:
        out = out + bias.reshape(-1, 1)
    return out


def swish(x: np.ndarray) -> np.ndarray:
    """Swish / SiLU. activation_type='swish' in the encoder config."""
    return (x * (1.0 / (1.0 + np.exp(-x)))).astype(np.float32)


def leaky_relu(x: np.ndarray, slope: float = 0.01) -> np.ndarray:
    return np.where(x >= 0.0, x, x * slope).astype(np.float32)


def softmax(x: np.ndarray, axis: int) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return (e / e.sum(axis=axis, keepdims=True)).astype(np.float32)


# ----------------------------------------------------------------------------
# Espnet relative positional encoding
# ----------------------------------------------------------------------------

def make_espnet_rel_pos_table(max_len: int = MAX_REL_LEN,
                                d_model: int = D_MODEL) -> np.ndarray:
    """Build the (1, 2*max_len-1, d_model) table that the rel-pos forward
    later indexes with center +- size + 1."""
    position = np.arange(0, max_len, dtype=np.float32).reshape(-1, 1)
    div_term = np.exp(np.arange(0, d_model, 2, dtype=np.float32) *
                       -(math.log(10000.0) / d_model))                # (d_model/2,)
    pe_positive = np.zeros((max_len, d_model), dtype=np.float32)
    pe_negative = np.zeros((max_len, d_model), dtype=np.float32)
    pe_positive[:, 0::2] = np.sin( position * div_term)
    pe_positive[:, 1::2] = np.cos( position * div_term)
    pe_negative[:, 0::2] = np.sin(-position * div_term)
    pe_negative[:, 1::2] = np.cos(-position * div_term)
    pe_positive = np.flip(pe_positive, axis=0)                        # (max_len, d_model), reversed
    pe = np.concatenate([pe_positive[None, :, :],
                         pe_negative[None, 1:, :]], axis=1)            # (1, 2*max_len-1, d_model)
    return pe.astype(np.float32)


def position_encoding(pe: np.ndarray, size: int) -> np.ndarray:
    """offset=0 case (the only one used in our path)."""
    L = pe.shape[1]
    return pe[:, L // 2 - size + 1 : L // 2 + size]                   # (1, 2*size-1, d_model)


# ----------------------------------------------------------------------------
# Sublayers
# ----------------------------------------------------------------------------

def linear_no_subsampling(x: np.ndarray, state: dict, prefix: str,
                            pos_pe_table: np.ndarray,
                            ) -> tuple[np.ndarray, np.ndarray]:
    """encoder.embed / encoder.up_embed forward.
    Returns (x_scaled, pos_emb).  x has been multiplied by sqrt(d_model)
    after the LN, matching EspnetRelPositionalEncoding.forward."""
    T = x.shape[1]
    x = linear(x,
                state[prefix + ".out.0.weight"],
                state[prefix + ".out.0.bias"])
    x = layer_norm(x,
                    state[prefix + ".out.1.weight"],
                    state[prefix + ".out.1.bias"],
                    eps=LN_EPS)
    x = x * math.sqrt(D_MODEL)                                          # xscale
    pos_emb = position_encoding(pos_pe_table, T)                        # (1, 2T-1, d)
    return x.astype(np.float32), pos_emb.astype(np.float32)


def pre_lookahead_layer(x: np.ndarray, state: dict, prefix: str) -> np.ndarray:
    """Causal lookahead conv block with residual. x: (B, T, C)."""
    B, T, C = x.shape
    assert B == 1
    inp = x
    # to channels-first
    h = x[0].T                                                           # (C, T)
    # right-pad by pre_lookahead_len
    h = np.pad(h, ((0, 0), (0, PRE_LOOKAHEAD_LEN)),
                mode='constant', constant_values=0.0)
    h = conv1d(h, state[prefix + ".conv1.weight"],
                state[prefix + ".conv1.bias"],
                stride=1, padding=0)
    h = leaky_relu(h)
    h = np.pad(h, ((0, 0), (2, 0)),
                mode='constant', constant_values=0.0)
    h = conv1d(h, state[prefix + ".conv2.weight"],
                state[prefix + ".conv2.bias"],
                stride=1, padding=0)
    # back to (B, T, C)
    h = h.T[None, :, :]
    return (h + inp).astype(np.float32)


def rel_shift(x: np.ndarray) -> np.ndarray:
    """x: (B, head, T, 2T-1) -> (B, head, T, T)."""
    B, H, T, L = x.shape
    assert L == 2 * T - 1
    zero_pad = np.zeros((B, H, T, 1), dtype=x.dtype)
    xp = np.concatenate([zero_pad, x], axis=-1)                          # (B, H, T, 2T)
    xp = xp.reshape(B, H, 2 * T, T)
    xp = xp[:, :, 1:, :].reshape(B, H, T, 2 * T - 1)
    return xp[:, :, :, :T].astype(np.float32)


def rel_pos_attn(x: np.ndarray, pos_emb: np.ndarray,
                  state: dict, prefix: str) -> np.ndarray:
    """RelPositionMultiHeadedAttention.forward (key_bias=True). x: (B, T, C)."""
    B, T, C = x.shape
    assert B == 1 and C == D_MODEL
    # qkv projections
    q = linear(x,
                state[prefix + ".linear_q.weight"],
                state[prefix + ".linear_q.bias"])                        # (B, T, C)
    k = linear(x,
                state[prefix + ".linear_k.weight"],
                state[prefix + ".linear_k.bias"])
    v = linear(x,
                state[prefix + ".linear_v.weight"],
                state[prefix + ".linear_v.bias"])
    q = q.reshape(B, T, N_HEADS, D_HEAD).transpose(0, 2, 1, 3)           # (B, H, T, d)
    k = k.reshape(B, T, N_HEADS, D_HEAD).transpose(0, 2, 1, 3)
    v = v.reshape(B, T, N_HEADS, D_HEAD).transpose(0, 2, 1, 3)
    # positional emb: linear_pos has no bias
    L = pos_emb.shape[1]                                                  # 2T - 1
    p = pos_emb @ state[prefix + ".linear_pos.weight"].T                  # (1, 2T-1, C)
    p = p.reshape(1, L, N_HEADS, D_HEAD).transpose(0, 2, 1, 3)            # (1, H, 2T-1, d)
    # bias_u/v are (H, d). Upstream code applies them on q in (B, T, H, d).
    q_t = q.transpose(0, 2, 1, 3)                                         # (B, T, H, d)
    q_u = (q_t + state[prefix + ".pos_bias_u"]).transpose(0, 2, 1, 3)     # (B, H, T, d)
    q_v = (q_t + state[prefix + ".pos_bias_v"]).transpose(0, 2, 1, 3)
    # matrix AC / BD
    ac = np.einsum('bhtd,bhsd->bhts', q_u, k)                             # (B, H, T, T)
    bd = np.einsum('bhtd,bhld->bhtl', q_v, p)                             # (B, H, T, 2T-1)
    if ac.shape != bd.shape:
        bd = rel_shift(bd)                                                # (B, H, T, T)
    scores = (ac + bd) / math.sqrt(D_HEAD)
    attn = softmax(scores, axis=-1)                                       # (B, H, T, T)
    ctx = np.einsum('bhts,bhsd->bhtd', attn, v)                           # (B, H, T, d)
    ctx = ctx.transpose(0, 2, 1, 3).reshape(B, T, C)                       # (B, T, C)
    return linear(ctx,
                   state[prefix + ".linear_out.weight"],
                   state[prefix + ".linear_out.bias"]).astype(np.float32)


def conformer_block(x: np.ndarray, pos_emb: np.ndarray,
                     state: dict, prefix: str) -> np.ndarray:
    """ConformerEncoderLayer with normalize_before=True, no macaron, no conv."""
    # MHA branch
    residual = x
    h = layer_norm(x,
                    state[prefix + ".norm_mha.weight"],
                    state[prefix + ".norm_mha.bias"],
                    eps=LN_EPS_ATTN)
    h = rel_pos_attn(h, pos_emb, state, prefix + ".self_attn")
    x = residual + h
    # FF branch
    residual = x
    h = layer_norm(x,
                    state[prefix + ".norm_ff.weight"],
                    state[prefix + ".norm_ff.bias"],
                    eps=LN_EPS_ATTN)
    h = linear(h,
                state[prefix + ".feed_forward.w_1.weight"],
                state[prefix + ".feed_forward.w_1.bias"])
    h = swish(h)
    h = linear(h,
                state[prefix + ".feed_forward.w_2.weight"],
                state[prefix + ".feed_forward.w_2.bias"])
    return (residual + h).astype(np.float32)


def up_layer(x: np.ndarray, state: dict, prefix: str) -> np.ndarray:
    """Upsample1D forward: nearest 2x interpolate -> left-pad 4 -> conv k=5.
    x: (B, T, C). Returns (B, 2T, C)."""
    B, T, C = x.shape
    assert B == 1
    h = x[0].T                                                            # (C, T)
    # nearest-2x upsample: each sample duplicated.
    h = np.repeat(h, UP_STRIDE, axis=1)                                   # (C, 2T)
    # left-pad by stride*2 = 4
    h = np.pad(h, ((0, 0), (UP_STRIDE * 2, 0)),
                mode='constant', constant_values=0.0)
    h = conv1d(h,
                state[prefix + ".conv.weight"],
                state[prefix + ".conv.bias"],
                stride=1, padding=0)
    return h.T[None, :, :].astype(np.float32)


# ----------------------------------------------------------------------------
# Full forward
# ----------------------------------------------------------------------------

def flow_encoder_forward(tokens: np.ndarray, spk_emb: np.ndarray,
                          state: dict, dumps: dict) -> tuple[np.ndarray, np.ndarray]:
    """tokens: (T,) int32 in [0, vocab_size).
       spk_emb: (192,) already L2-normalized.
       Returns: (1, 2T, 80) encoder output, (80,) spk_affine output.
       Fills `dumps` with named stage tensors for the C++ test."""
    # ---- speaker affine ----
    spk = linear(spk_emb[None, :],                                          # (1, 192)
                  state["flow.spk_embed_affine_layer.weight"],
                  state["flow.spk_embed_affine_layer.bias"])[0]              # (80,)

    # ---- input embedding ----
    emb_w = state["flow.input_embedding.weight"]                              # (6561, 512)
    x = emb_w[tokens][None, :, :].astype(np.float32)                          # (1, T, 512)
    dumps["after_input_embedding"] = x.copy()

    # ---- pre-stack #1 ----
    pos_pe_table = make_espnet_rel_pos_table(MAX_REL_LEN, D_MODEL)
    x, pos_emb = linear_no_subsampling(x, state, "flow.encoder.embed",
                                         pos_pe_table)
    dumps["after_embed"] = x.copy()

    x = pre_lookahead_layer(x, state, "flow.encoder.pre_lookahead_layer")
    dumps["after_prelookahead"] = x.copy()

    for i in range(N_BLOCKS):
        x = conformer_block(x, pos_emb, state, f"flow.encoder.encoders.{i}")
        if i == 0:
            dumps["after_enc_block0"] = x.copy()
    dumps["after_enc_block5"] = x.copy()

    # ---- upsample ----
    x = up_layer(x, state, "flow.encoder.up_layer")                            # (1, 2T, 512)
    dumps["after_uplayer"] = x.copy()

    x, pos_emb = linear_no_subsampling(x, state, "flow.encoder.up_embed",
                                         pos_pe_table)
    dumps["after_upembed"] = x.copy()

    for i in range(N_UP_BLOCKS):
        x = conformer_block(x, pos_emb, state, f"flow.encoder.up_encoders.{i}")
        if i == 0:
            dumps["after_upenc_block0"] = x.copy()
    dumps["after_upenc_block3"] = x.copy()

    # ---- final norm + projection ----
    x = layer_norm(x,
                    state["flow.encoder.after_norm.weight"],
                    state["flow.encoder.after_norm.bias"],
                    eps=LN_EPS)
    dumps["after_afternorm"] = x.copy()

    x = linear(x,
                state["flow.encoder_proj.weight"],
                state["flow.encoder_proj.bias"])                                # (1, 2T, 80)
    dumps["after_encoderproj"] = x.copy()

    return x.astype(np.float32), spk.astype(np.float32)


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------

def make_test_inputs(T: int, seed: int = 31
                      ) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    tokens = rng.randint(0, VOCAB_SIZE, size=(T,)).astype(np.int32)
    spk    = rng.standard_normal(SPK_DIM).astype(np.float32)
    # L2-normalize, matching F.normalize(embedding, dim=1).
    spk = spk / max(np.linalg.norm(spk), 1e-12)
    return tokens, spk.astype(np.float32)


def write_stage_dump(f, name: str, tensor: np.ndarray) -> None:
    nb = name.encode("ascii")
    assert len(nb) <= 31
    f.write(nb + b"\x00" * (32 - len(nb)))
    # Flatten to (1, T, C) layout; allow 1D or 2D inputs to be promoted.
    if tensor.ndim == 2:
        tensor = tensor[None, :, :]
    assert tensor.ndim == 3, f"{name}: expected 3D, got {tensor.shape}"
    d0, d1, d2 = tensor.shape
    f.write(np.int32(d0).tobytes())
    f.write(np.int32(d1).tobytes())
    f.write(np.int32(d2).tobytes())
    f.write(tensor.astype(np.float32).tobytes())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-bin",  default="tests/flow_encoder_reference.bin")
    ap.add_argument("--out-meta", default="tests/flow_encoder_reference.json")
    ap.add_argument("--T",        type=int, default=16,
                    help="number of speech tokens in the test input")
    args = ap.parse_args()

    print(f"Fetching {S3GEN_FILE} ...")
    src = Path(hf_hub_download(REPO_ID, S3GEN_FILE))
    state_full = load_file(str(src))

    # Keep just the flow.* tensors; everything else (mel2wav, speaker_encoder,
    # tokenizer, decoder) is irrelevant to this oracle.
    state = {k: v.astype(np.float32) for k, v in state_full.items()
              if k.startswith("flow.")}
    print(f"  flow.* tensors: {len(state)}")

    tokens, spk = make_test_inputs(args.T)
    print(f"tokens: {tokens.shape}  range=[{tokens.min()}, {tokens.max()}]")
    print(f"spk:    {spk.shape}     L2={np.linalg.norm(spk):.4f}")

    dumps: dict[str, np.ndarray] = {}
    out, spk_aff = flow_encoder_forward(tokens, spk, state, dumps)
    print(f"\nencoder_proj output: {out.shape}  "
          f"min={float(out.min()):+.4f}  max={float(out.max()):+.4f}  "
          f"avg={float(out.mean()):+.4f}")
    print(f"spk_affine output:   {spk_aff.shape}  "
          f"min={float(spk_aff.min()):+.4f}  max={float(spk_aff.max()):+.4f}  "
          f"avg={float(spk_aff.mean()):+.4f}")

    print("\nStage dumps:")
    for k, t in dumps.items():
        print(f"  {k:24s} shape={t.shape}  "
              f"min={float(t.min()):+.4f}  max={float(t.max()):+.4f}")

    out_bin = Path(args.out_bin)
    out_bin.parent.mkdir(parents=True, exist_ok=True)
    with open(out_bin, "wb") as f:
        f.write(np.int32(args.T).tobytes())          # T_in
        f.write(np.int32(D_MODEL).tobytes())          # d_in (encoder hidden)
        f.write(np.int32(D_OUT).tobytes())            # d_out
        f.write(np.int32(len(dumps)).tobytes())       # n_stages
        f.write(tokens.tobytes())                     # int32 tokens
        f.write(spk.tobytes())                        # fp32 (192,)
        f.write(spk_aff.tobytes())                    # fp32 (80,)
        for name, t in dumps.items():
            write_stage_dump(f, name, t)

    Path(args.out_meta).write_text(json.dumps({
        "T_in":      int(args.T),
        "T_out":     int(out.shape[1]),
        "d_in":      D_MODEL,
        "d_out":     D_OUT,
        "vocab":     VOCAB_SIZE,
        "n_stages":  len(dumps),
        "stages":    list(dumps.keys()),
        "out_min":   float(out.min()),
        "out_max":   float(out.max()),
        "out_mean":  float(out.mean()),
        "spk_aff_min":  float(spk_aff.min()),
        "spk_aff_max":  float(spk_aff.max()),
        "spk_aff_mean": float(spk_aff.mean()),
    }, indent=2), encoding="utf-8")

    print(f"\nWrote {out_bin} ({out_bin.stat().st_size} bytes)")
    print(f"Wrote {args.out_meta}")


if __name__ == "__main__":
    main()
