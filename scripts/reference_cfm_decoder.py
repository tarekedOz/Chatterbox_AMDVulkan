"""
NumPy reference for the S3Gen CFM decoder (CausalConditionalCFM +
ConditionalDecoder, the UNet noise predictor).

Mirrors upstream chatterbox/models/s3gen/{decoder, flow_matching}.py and
the matcha primitives (decoder.py, transformer.py) used by Turbo with
meanflow=True, channels=[256], n_blocks=4, num_mid_blocks=12.

UNet (single down/up stage; no actual time-domain sub/upsampling
since channels=[256] means is_last=True on the only stage):

    x  (1, 80, T) noised mel
    mu (1, 80, T) encoder output (already at mel rate)
    spks (1, 80)  speaker-affine output
    cond (1, 80, T) prompt-feat conditioning (zero-padded for the
                    generated portion)
      |
      pack -> (1, 320, T)
      |
    Down block 0:
      CausalResnetBlock1D(320 -> 256)
        block1: CausalConv1d(k=3) -> LN -> Mish
        + time_mlp(t_emb) injection
        block2: CausalConv1d(k=3) -> LN -> Mish
        + res_conv(320 -> 256, k=1) skip
      4x BasicTransformerBlock(dim=256, 8 heads x 64 head_dim,
                                LN + self-attn + LN + GELU-FF 256->1024->256)
      CausalConv1d(256 -> 256, k=3)   <-- "downsample" but no stride change
                                          since is_last=True
      [save x as skip]
      |
    Mid blocks 0..11 (12 total):
      CausalResnetBlock1D(256 -> 256)
      4x BasicTransformerBlock
      |
    Up block 0:
      x = cat([x[:, :, :skip.shape[-1]], skip], dim=1)  -> (1, 512, T)
      CausalResnetBlock1D(512 -> 256, with res_conv 512 -> 256)
      4x BasicTransformerBlock
      CausalConv1d(256 -> 256, k=3)   <-- "upsample" but no stride change
      |
    CausalBlock1D(256 -> 256)         (final_block)
    Conv1d(256 -> 80, k=1)             (final_proj)
      |
    * mask                              (zero out padded positions)
    return dxdt (1, 80, T)

Time embedding:
    t_emb = SinusoidalPosEmb(in_channels=320)(t * scale=1000)
        emb = log(10000) / (160 - 1)
        emb = exp(arange(160) * -emb)
        emb = scale * t * emb       (shape (1, 160))
        emb = cat(sin(emb), cos(emb), -1)   (1, 320)
    t_emb = TimestepEmbedding(t_emb)
        Linear(320 -> 1024) -> silu -> Linear(1024 -> 1024)
    If meanflow:
        r_emb = same pipeline on r
        concat = cat([t_emb, r_emb], dim=1)   (1, 2048)
        t_emb = time_embed_mixer(concat)       Linear(2048 -> 1024, no bias)

CFM solver (meanflow, basic_euler):
    z = N(0, 1) noise of shape (1, 80, T)
    t_span = linspace(0, 1, n_timesteps + 1)   (no cosine, since meanflow)
    For each (t, r) in zip(t_span[:-1], t_span[1:]):
        dxdt = decoder(z, mask, mu, t, spks, cond, r)
        z = z + (r - t) * dxdt
    Return z

Output: tests/cfm_decoder_reference.bin
    int32  T_mel
    int32  d_mel = 80
    int32  d_in  = 320
    int32  d_time = 1024
    int32  n_stages
    int32  n_timesteps = 2

    float32[T_mel * 80]      mu                (1, 80, T) (mu input)
    float32[80]              spks
    float32[T_mel * 80]      cond
    float32[T_mel]           mask              (1, 1, T) — all 1s
    float32                  t                 single scalar t for the
                                                estimator-only stage dumps
    float32                  r                 same for r
    float32[T_mel * 80]      x_in              fresh-noise input z for the
                                                estimator-only stage dumps

    n_stages * (name + tensor):
        char[32]   stage_name
        int32      dim0=1
        int32      dim1  ((T_mel for spatial tensors, 1 for vectors))
        int32      dim2  ((channel dim))
        ((then float32 data of dim0*dim1*dim2))

    Then the SOLVER output:
        int32      solver_T = T_mel
        float32[T_mel * 80]    solver_z_init      initial noise (same as x_in above)
        float32[T_mel * 80]    solver_final       final mel (1, 80, T)
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

# --- Architecture constants (Turbo, meanflow) ---
IN_CHANNELS      = 320     # x(80) + mu(80) + spks(80) + cond(80)
OUT_CHANNELS     = 80
TIME_EMB_IN      = 320     # SinusoidalPosEmb dim (= in_channels for decoder)
TIME_EMB_DIM     = 1024    # channels[0] * 4 = 256 * 4
MID_CHANNELS     = 256
SKIP_UP_IN       = 512     # mid_channels * 2 (skip concat)
N_HEADS          = 8
HEAD_DIM         = 64
ATTN_INNER       = N_HEADS * HEAD_DIM   # 512
FF_HIDDEN        = MID_CHANNELS * 4     # 1024
N_TRANSFORMERS   = 4
N_MID_BLOCKS     = 12
LN_EPS           = 1e-5
N_TIMESTEPS      = 2       # meanflow default for Turbo


# ---------------------------------------------------------------------------
# Activations / primitives
# ---------------------------------------------------------------------------

def silu(x: np.ndarray) -> np.ndarray:
    return (x * (1.0 / (1.0 + np.exp(-x)))).astype(np.float32)


def mish(x: np.ndarray) -> np.ndarray:
    # x * tanh(softplus(x)) = x * tanh(log(1 + exp(x)))
    # Numerically stable softplus.
    sp = np.where(x > 20.0, x, np.log1p(np.exp(np.minimum(x, 20.0))))
    return (x * np.tanh(sp)).astype(np.float32)


def gelu(x: np.ndarray) -> np.ndarray:
    # Exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    return (0.5 * x * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))
             ).astype(np.float32)


def softmax(x: np.ndarray, axis: int) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return (e / e.sum(axis=axis, keepdims=True)).astype(np.float32)


def linear(x: np.ndarray, w: np.ndarray,
            b: np.ndarray | None = None) -> np.ndarray:
    """torch.nn.Linear forward. w: (out, in). x last dim is in."""
    y = x @ w.T
    if b is not None:
        y = y + b
    return y.astype(np.float32)


def layer_norm(x: np.ndarray, w: np.ndarray, b: np.ndarray,
                eps: float = LN_EPS) -> np.ndarray:
    """LayerNorm over the last axis."""
    mean = x.mean(axis=-1, keepdims=True)
    var  = x.var (axis=-1, keepdims=True)
    y = (x - mean) / np.sqrt(var + eps)
    return (y * w + b).astype(np.float32)


def conv1d(x: np.ndarray, w: np.ndarray, b: np.ndarray | None,
            stride: int = 1, padding: int = 0,
            dilation: int = 1) -> np.ndarray:
    """x: (C_in, T). w: (C_out, C_in, K). Returns (C_out, T_out)."""
    Cin, T = x.shape
    Cout, _, K = w.shape
    xp = np.pad(x, ((0, 0), (padding, padding)))
    Tp = xp.shape[1]
    eff_K = (K - 1) * dilation + 1
    T_out = (Tp - eff_K) // stride + 1
    col = np.empty((Cin * K, T_out), dtype=np.float32)
    for k in range(K):
        start = k * dilation
        end = start + stride * T_out
        col[k * Cin:(k + 1) * Cin, :] = xp[:, start:end:stride]
    w_flat = w.transpose(2, 1, 0).reshape(K * Cin, Cout)
    out = w_flat.T @ col
    out = out.reshape(Cout, T_out).astype(np.float32)
    if b is not None:
        out = out + b.reshape(-1, 1)
    return out


def causal_conv1d(x: np.ndarray, w: np.ndarray, b: np.ndarray | None,
                    K: int) -> np.ndarray:
    """x: (C_in, T). Pad LEFT by K-1 then standard conv. Returns (C_out, T)."""
    Cin, T = x.shape
    xp = np.pad(x, ((0, 0), (K - 1, 0)))
    return conv1d(xp, w, b, stride=1, padding=0)


# ---------------------------------------------------------------------------
# Sub-layers
# ---------------------------------------------------------------------------

def sinusoidal_pos_emb(t: float, dim: int = TIME_EMB_IN,
                        scale: float = 1000.0) -> np.ndarray:
    """t: scalar. Returns (1, dim) numpy array."""
    half = dim // 2
    emb = math.log(10000.0) / (half - 1)
    emb = np.exp(np.arange(half, dtype=np.float32) * -emb)
    emb = scale * t * emb                          # (half,)
    return np.concatenate([np.sin(emb), np.cos(emb)]).reshape(1, dim).astype(np.float32)


def timestep_embedding(t_emb: np.ndarray, state: dict,
                        prefix: str = "flow.decoder.estimator.time_mlp"
                        ) -> np.ndarray:
    """Linear(in -> hidden) -> silu -> Linear(hidden -> hidden)."""
    x = linear(t_emb,
                state[prefix + ".linear_1.weight"],
                state[prefix + ".linear_1.bias"])
    x = silu(x)
    x = linear(x,
                state[prefix + ".linear_2.weight"],
                state[prefix + ".linear_2.bias"])
    return x.astype(np.float32)


def causal_block1d(x: np.ndarray, mask: np.ndarray, state: dict,
                    prefix: str) -> np.ndarray:
    """x: (B=1, C, T), mask: (1, 1, T).
       block.0 = CausalConv1d(k=3), block.2 = LayerNorm, block.4 = Mish."""
    x = x * mask                                                  # x * mask
    w  = state[prefix + ".block.0.weight"]
    bC = state[prefix + ".block.0.bias"]
    h  = causal_conv1d(x[0], w, bC, K=3)[None, :, :]              # (1, Cout, T)
    # Transpose to (1, T, C) for LayerNorm over C, then transpose back.
    h_TC = h.transpose(0, 2, 1)                                    # (1, T, C)
    h_TC = layer_norm(h_TC,
                       state[prefix + ".block.2.weight"],
                       state[prefix + ".block.2.bias"])
    h = h_TC.transpose(0, 2, 1)                                    # (1, C, T)
    h = mish(h)
    return (h * mask).astype(np.float32)


def causal_resnet_block1d(x: np.ndarray, mask: np.ndarray,
                           t_emb: np.ndarray, state: dict,
                           prefix: str) -> np.ndarray:
    """ResnetBlock1D with two CausalBlock1Ds + time MLP between them +
       1x1 res_conv on the masked input."""
    h = causal_block1d(x, mask, state, prefix + ".block1")          # (1, Cout, T)
    # time MLP: Sequential(Mish, Linear(t_emb_dim, dim_out)).
    tm = mish(t_emb)
    tm = linear(tm,
                 state[prefix + ".mlp.1.weight"],
                 state[prefix + ".mlp.1.bias"])                       # (1, Cout)
    h  = h + tm[:, :, None]                                          # broadcast over T
    h  = causal_block1d(h, mask, state, prefix + ".block2")
    # res_conv: 1x1 Conv1d on (x * mask)
    rw = state[prefix + ".res_conv.weight"]
    rb = state[prefix + ".res_conv.bias"]
    sc = conv1d((x * mask)[0], rw, rb, stride=1, padding=0)[None, :, :]
    return (h + sc).astype(np.float32)


def basic_transformer_block(x_BTC: np.ndarray, attn_mask_bias: np.ndarray,
                              state: dict, prefix: str) -> np.ndarray:
    """BasicTransformerBlock forward.
       x_BTC: (B=1, T, C=256). attn_mask_bias: (B=1, T, T) additive bias
       (0 where valid, -1e10 elsewhere)."""
    B, T, C = x_BTC.shape
    # ---- 1. Self-attn ----
    h = layer_norm(x_BTC,
                    state[prefix + ".norm1.weight"],
                    state[prefix + ".norm1.bias"])
    q = linear(h, state[prefix + ".attn1.to_q.weight"])   # no bias  (1, T, 512)
    k = linear(h, state[prefix + ".attn1.to_k.weight"])
    v = linear(h, state[prefix + ".attn1.to_v.weight"])
    q = q.reshape(B, T, N_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)     # (B, H, T, Hd)
    k = k.reshape(B, T, N_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
    v = v.reshape(B, T, N_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
    scores = np.einsum('bhtd,bhsd->bhts', q, k) / math.sqrt(HEAD_DIM)
    # Add attention mask bias (broadcast over heads).
    if attn_mask_bias is not None:
        scores = scores + attn_mask_bias[:, None, :, :]
    attn = softmax(scores, axis=-1)
    ctx  = np.einsum('bhts,bhsd->bhtd', attn, v)                     # (B, H, T, Hd)
    ctx  = ctx.transpose(0, 2, 1, 3).reshape(B, T, ATTN_INNER)
    attn_out = linear(ctx,
                       state[prefix + ".attn1.to_out.0.weight"],
                       state[prefix + ".attn1.to_out.0.bias"])
    x_BTC = x_BTC + attn_out

    # ---- 2. FF (no cross-attn) ----
    h = layer_norm(x_BTC,
                    state[prefix + ".norm3.weight"],
                    state[prefix + ".norm3.bias"])
    # ff.net.0 = GELU(dim_in, dim_out) — has .proj Linear, then GELU activation.
    h = linear(h,
                state[prefix + ".ff.net.0.proj.weight"],
                state[prefix + ".ff.net.0.proj.bias"])               # (1, T, 1024)
    h = gelu(h)
    h = linear(h,
                state[prefix + ".ff.net.2.weight"],
                state[prefix + ".ff.net.2.bias"])                     # (1, T, 256)
    return (x_BTC + h).astype(np.float32)


def transformer_stack(x_BCT: np.ndarray, mask: np.ndarray, t_emb: np.ndarray,
                       state: dict, prefix: str, n_blocks: int
                       ) -> np.ndarray:
    """Rearrange (B, C, T) -> (B, T, C); run n_blocks BasicTransformerBlocks;
       rearrange back. attention_mask is built from `mask` (all 1s -> zeros).
       t_emb is unused inside the transformer (no AdaLayerNorm) but kept for
       interface symmetry with upstream."""
    B, C, T = x_BCT.shape
    x_BTC = x_BCT.transpose(0, 2, 1)                                  # (B, T, C)
    # attention mask bias: (B, T, T). Built by add_optional_chunk_mask
    # with full-attn settings (chunk_size=0, num_decoding_left_chunks=-1).
    # That reduces to attn_mask = mask^T @ mask (broadcast pairwise), bool;
    # then mask_to_bias converts to additive (0 / -1e10).
    m_BT = mask[:, 0, :]                                              # (B, T) float
    attn_bool = (m_BT[:, :, None] * m_BT[:, None, :]) > 0              # (B, T, T) bool
    attn_bias = np.where(attn_bool, 0.0, -1.0e10).astype(np.float32)
    for i in range(n_blocks):
        x_BTC = basic_transformer_block(x_BTC, attn_bias, state,
                                          f"{prefix}.{i}")
    return x_BTC.transpose(0, 2, 1).astype(np.float32)               # (B, C, T)


# ---------------------------------------------------------------------------
# Full estimator forward
# ---------------------------------------------------------------------------

def estimator_forward(x: np.ndarray, mask: np.ndarray, mu: np.ndarray,
                       t: float, spks: np.ndarray, cond: np.ndarray,
                       r: float, state: dict, dumps: dict) -> np.ndarray:
    """
    x:    (1, 80, T)
    mask: (1, 1, T)
    mu:   (1, 80, T)
    t:    scalar
    spks: (1, 80)
    cond: (1, 80, T)
    r:    scalar (meanflow end time)
    """
    B, _, T = x.shape
    assert B == 1

    # ---- Time embedding ----
    t_emb = sinusoidal_pos_emb(t)                                     # (1, 320)
    t_emb = timestep_embedding(t_emb, state)                          # (1, 1024)
    dumps["after_t_emb"] = t_emb.copy().reshape(1, 1, TIME_EMB_DIM)

    r_emb = sinusoidal_pos_emb(r)
    r_emb = timestep_embedding(r_emb, state)                          # (1, 1024)
    dumps["after_r_emb"] = r_emb.copy().reshape(1, 1, TIME_EMB_DIM)

    concat = np.concatenate([t_emb, r_emb], axis=1)                   # (1, 2048)
    t_emb = linear(concat,
                    state["flow.decoder.estimator.time_embed_mixer.weight"],
                    None)                                              # no bias
    dumps["after_time_mixer"] = t_emb.copy().reshape(1, 1, TIME_EMB_DIM)

    # ---- Pack channels ----
    # x, mu (1, 80, T) + spks (1, 80) repeated to (1, 80, T) + cond (1, 80, T)
    spks_BCT = np.broadcast_to(spks[:, :, None], (B, OUT_CHANNELS, T)
                                ).astype(np.float32)
    packed = np.concatenate([x, mu, spks_BCT, cond], axis=1)          # (1, 320, T)
    dumps["after_pack"] = packed.copy()

    # ---- Down block 0 ----
    pfx = "flow.decoder.estimator.down_blocks.0"
    x = causal_resnet_block1d(packed, mask, t_emb, state, pfx + ".0")  # (1, 256, T)
    dumps["after_down0_resnet"] = x.copy()
    x = transformer_stack(x, mask, t_emb, state, pfx + ".1", N_TRANSFORMERS)
    dumps["after_down0_transformer"] = x.copy()
    skip = x.copy()                                                    # save for up-block
    # downsample = CausalConv1d(256, 256, k=3) — no actual subsampling
    w  = state[pfx + ".2.weight"]
    bb = state[pfx + ".2.bias"]
    x = causal_conv1d((x * mask)[0], w, bb, K=3)[None, :, :]
    dumps["after_down0_downsample"] = x.copy()

    # ---- Mid blocks ----
    for i in range(N_MID_BLOCKS):
        mpfx = f"flow.decoder.estimator.mid_blocks.{i}"
        x = causal_resnet_block1d(x, mask, t_emb, state, mpfx + ".0")
        x = transformer_stack(x, mask, t_emb, state, mpfx + ".1",
                               N_TRANSFORMERS)
        if i == 0:                   dumps["after_mid0"] = x.copy()
        if i == N_MID_BLOCKS - 1:    dumps["after_mid11"] = x.copy()

    # ---- Up block 0 ----
    # Concat with skip along channel dim.
    upfx = "flow.decoder.estimator.up_blocks.0"
    x_cat = np.concatenate([x[:, :, :skip.shape[-1]], skip], axis=1)   # (1, 512, T)
    x = causal_resnet_block1d(x_cat, mask, t_emb, state, upfx + ".0")
    x = transformer_stack(x, mask, t_emb, state, upfx + ".1",
                           N_TRANSFORMERS)
    dumps["after_up0_transformer"] = x.copy()
    w  = state[upfx + ".2.weight"]
    bb = state[upfx + ".2.bias"]
    x = causal_conv1d((x * mask)[0], w, bb, K=3)[None, :, :]
    dumps["after_up0_upsample"] = x.copy()

    # ---- Final block + projection ----
    x = causal_block1d(x, mask, state,
                         "flow.decoder.estimator.final_block")
    dumps["after_final_block"] = x.copy()
    w  = state["flow.decoder.estimator.final_proj.weight"]   # (80, 256, 1)
    bb = state["flow.decoder.estimator.final_proj.bias"]
    x = conv1d((x * mask)[0], w, bb, stride=1, padding=0)[None, :, :]
    x = x * mask
    dumps["after_final_proj"] = x.copy()
    return x.astype(np.float32)


# ---------------------------------------------------------------------------
# Meanflow CFM solver (basic_euler)
# ---------------------------------------------------------------------------

def meanflow_basic_euler(z: np.ndarray, mu: np.ndarray, mask: np.ndarray,
                          spks: np.ndarray, cond: np.ndarray, state: dict,
                          n_timesteps: int = N_TIMESTEPS) -> np.ndarray:
    """z: (1, 80, T). Returns mel (1, 80, T) after n_timesteps Euler steps."""
    t_span = np.linspace(0.0, 1.0, n_timesteps + 1, dtype=np.float32)
    x = z.copy()
    for ti in range(n_timesteps):
        t = float(t_span[ti])
        r = float(t_span[ti + 1])
        dumps_throwaway: dict = {}
        dxdt = estimator_forward(x, mask, mu, t, spks, cond, r, state,
                                   dumps_throwaway)
        dt = r - t
        x = (x + dt * dxdt).astype(np.float32)
    return x


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def make_test_inputs(T_mel: int, seed: int = 41
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    mu   = rng.standard_normal((1, OUT_CHANNELS, T_mel)).astype(np.float32) * 0.5
    spks = rng.standard_normal((1, OUT_CHANNELS)).astype(np.float32) * 0.2
    cond = np.zeros((1, OUT_CHANNELS, T_mel), dtype=np.float32)
    # cond's first N/3 frames are real (the prompt prefix), the rest are
    # zero. Mimics upstream pattern: cond[:, :, :prompt_len] = prompt_feat.
    prompt = T_mel // 3
    cond[:, :, :prompt] = rng.standard_normal((1, OUT_CHANNELS, prompt)
                                                ).astype(np.float32) * 0.3
    mask = np.ones((1, 1, T_mel), dtype=np.float32)
    return mu, spks, cond, mask


def write_stage(f, name: str, t: np.ndarray) -> None:
    """Tensors are dumped in (1, T, C) numpy layout so that the raw bytes
    match ggml ne=(C, T) directly. Time axis = numpy axis 1, channel axis
    = numpy axis 2 (innermost)."""
    nb = name.encode("ascii")
    assert len(nb) <= 31
    f.write(nb + b"\x00" * (32 - len(nb)))
    if t.ndim == 2:
        t = t[None, :, :]
    assert t.ndim == 3, f"{name}: expected 3D, got {t.shape}"
    # Internal compute keeps (B, C, T) (PyTorch convention). Transpose
    # to (B, T, C) before dumping so bytes match ggml (C, T) ne ordering.
    # For pure vectors (B=1, T=1, C=*) the transpose is a no-op.
    if t.shape[1] != 1 and t.shape[2] != 1:
        t = t.transpose(0, 2, 1)
    elif t.shape[1] == 1:
        # (B, 1, C) — pure feature vector. Keep as is.
        pass
    d0, d1, d2 = t.shape
    f.write(np.int32(d0).tobytes())
    f.write(np.int32(d1).tobytes())
    f.write(np.int32(d2).tobytes())
    f.write(np.ascontiguousarray(t.astype(np.float32)).tobytes())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-bin",  default="tests/cfm_decoder_reference.bin")
    ap.add_argument("--out-meta", default="tests/cfm_decoder_reference.json")
    ap.add_argument("--T-mel",    type=int, default=16,
                    help="Number of mel frames in the test input.")
    args = ap.parse_args()

    print(f"Fetching {S3GEN_FILE} ...")
    src = Path(hf_hub_download(REPO_ID, S3GEN_FILE))
    state_full = load_file(str(src))
    state = {k: v.astype(np.float32) for k, v in state_full.items()
              if k.startswith("flow.decoder.")}
    print(f"  flow.decoder.* tensors: {len(state)}")

    T_mel = args.T_mel
    mu, spks, cond, mask = make_test_inputs(T_mel)
    # Deterministic noise for the estimator/solver tests.
    z = np.random.RandomState(7).standard_normal((1, OUT_CHANNELS, T_mel)
                                                   ).astype(np.float32)

    # Probe time-step: pick the first non-zero (t=0.0 would skip the
    # update) — use the mid-point of the meanflow schedule.
    t_probe = 0.5
    r_probe = 1.0

    print(f"\n[1/2] Estimator forward at t={t_probe}, r={r_probe}, T={T_mel}")
    dumps: dict[str, np.ndarray] = {}
    dxdt = estimator_forward(z, mask, mu, t_probe, spks, cond, r_probe,
                              state, dumps)
    print(f"  dxdt: {dxdt.shape}  "
          f"min={float(dxdt.min()):+.4f}  max={float(dxdt.max()):+.4f}  "
          f"avg={float(dxdt.mean()):+.4f}")
    for k, v in dumps.items():
        print(f"  {k:30s} shape={v.shape}  "
              f"min={float(v.min()):+.4f}  max={float(v.max()):+.4f}")

    print(f"\n[2/2] Meanflow basic_euler (n_timesteps={N_TIMESTEPS})")
    out = meanflow_basic_euler(z, mu, mask, spks, cond, state)
    print(f"  output: {out.shape}  "
          f"min={float(out.min()):+.4f}  max={float(out.max()):+.4f}  "
          f"avg={float(out.mean()):+.4f}")

    # Write binary. All 3D (B, C, T) numpy tensors are transposed to
    # (B, T, C) so that the raw bytes match ggml ne=(C, T) directly.
    def to_TC(t: np.ndarray) -> np.ndarray:
        # (1, C, T) -> (T, C) contiguous
        return np.ascontiguousarray(t[0].T.astype(np.float32))

    out_bin = Path(args.out_bin)
    out_bin.parent.mkdir(parents=True, exist_ok=True)
    with open(out_bin, "wb") as f:
        f.write(np.int32(T_mel).tobytes())
        f.write(np.int32(OUT_CHANNELS).tobytes())
        f.write(np.int32(IN_CHANNELS).tobytes())
        f.write(np.int32(TIME_EMB_DIM).tobytes())
        f.write(np.int32(len(dumps)).tobytes())
        f.write(np.int32(N_TIMESTEPS).tobytes())

        f.write(to_TC(mu).tobytes())                       # (T, 80)
        f.write(spks[0].astype(np.float32).tobytes())      # (80,)
        f.write(to_TC(cond).tobytes())                     # (T, 80)
        f.write(mask[0, 0].astype(np.float32).tobytes())   # (T,)
        f.write(np.float32(t_probe).tobytes())
        f.write(np.float32(r_probe).tobytes())
        f.write(to_TC(z).tobytes())                         # (T, 80)

        for name, t in dumps.items():
            write_stage(f, name, t)

        # Solver section.
        f.write(np.int32(T_mel).tobytes())
        f.write(to_TC(z).tobytes())                         # init noise (T, 80)
        f.write(to_TC(out).tobytes())                       # final mel (T, 80)

    Path(args.out_meta).write_text(json.dumps({
        "T_mel":      int(T_mel),
        "d_mel":      OUT_CHANNELS,
        "d_in":       IN_CHANNELS,
        "d_time":     TIME_EMB_DIM,
        "n_stages":   len(dumps),
        "stages":     list(dumps.keys()),
        "n_timesteps":N_TIMESTEPS,
        "t_probe":    t_probe,
        "r_probe":    r_probe,
        "dxdt_min":   float(dxdt.min()),
        "dxdt_max":   float(dxdt.max()),
        "out_min":    float(out.min()),
        "out_max":    float(out.max()),
        "out_mean":   float(out.mean()),
    }, indent=2), encoding="utf-8")
    print(f"\nWrote {out_bin} ({out_bin.stat().st_size} bytes)")
    print(f"Wrote {args.out_meta}")


if __name__ == "__main__":
    main()
