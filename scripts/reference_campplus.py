"""
NumPy reference for the CAMPPlus speaker encoder.

Mirrors upstream chatterbox/models/s3gen/xvector.py exactly. Used as
the parity oracle for the C++ port (chatterbox-cpp/src/s3spk.{h,cpp},
to be written in the next session).

Pipeline:

    fbank (T, 80)                          (Kaldi-fbank 80-d input)
      -> permute -> (80, T)
      -> FCM head (Conv2d-on-(1,80,T) with 3 stride-2 ResNet stages
                   over the FREQ dim only; reshape to flatten)
                                            -> (320, T)
      -> TDNNLayer(320 -> 128, k=5, s=2)   -> (128, T/2)
      -> CAMDenseTDNNBlock(num_layers=12, growth=32, k=3, d=1)
         (DenseNet concat: 128 + 12*32 = 512)
      -> TransitLayer(512 -> 256)
      -> CAMDenseTDNNBlock(num_layers=24, growth=32, k=3, d=2)
         (256 + 24*32 = 1024)
      -> TransitLayer(1024 -> 512)
      -> CAMDenseTDNNBlock(num_layers=16, growth=32, k=3, d=2)
         (512 + 16*32 = 1024)
      -> TransitLayer(1024 -> 512)
      -> out_nonlinear (BN + ReLU)
      -> StatsPool (mean+std over time)    -> (1024,)
      -> DenseLayer(1024 -> 192, config_str="batchnorm_")
         (1x1 conv + BN without affine — the only place affine=False)
                                            -> (192,)

The output is NOT L2-normalized here; L2 happens later in
CausalMaskedDiffWithXvec.

Each "BatchNorm + Activation" inside the layers is per-channel:
  y = (x - running_mean) / sqrt(running_var + eps) * weight + bias  (then ReLU)

Tensor name conventions in our GGUF (under speaker_encoder.*):
  head.{conv1, bn1, layer1, layer2, conv2, bn2}    — FCM
  xvector.tdnn.{linear, nonlinear.batchnorm}       — TDNN
  xvector.block{1,2,3}.tdnnd{1..N}.{...}            — DenseTDNN sublayers
  xvector.transit{1,2,3}.{linear, nonlinear.batchnorm}
  xvector.out_nonlinear.batchnorm
  xvector.dense.{linear, nonlinear.batchnorm}

Output: tests/campplus_reference.bin (binary)
    int32   T               fbank frame count
    int32   n_mels          80
    float32 * (T * n_mels)  fbank input (T, 80) row-major
    int32   emb_dim         192
    float32 * emb_dim       expected unnormalized embedding
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

REPO_ID    = "ResembleAI/chatterbox-turbo"
S3GEN_FILE = "s3gen_meanflow.safetensors"

FEAT_DIM       = 80
EMB_DIM        = 192
GROWTH         = 32
BN_SIZE        = 4               # bn_channels = bn_size * growth = 128
INIT_CHANNELS  = 128
SEG_LEN        = 100             # CAMLayer seg_pooling kernel
BN_EPS         = 1e-5

BLOCK_SPECS = [
    # (num_layers, kernel_size, dilation)
    (12, 3, 1),
    (24, 3, 2),
    (16, 3, 2),
]


# ---------------------------------------------------------------------------
# Primitive helpers
# ---------------------------------------------------------------------------

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def bn(x: np.ndarray, rm: np.ndarray, rv: np.ndarray,
       w: np.ndarray | None, b: np.ndarray | None,
       eps: float = BN_EPS) -> np.ndarray:
    """Per-channel BatchNorm inference. x has channels on axis 0
    (any number of trailing dims). w, b may be None for affine=False."""
    shape = (-1,) + (1,) * (x.ndim - 1)
    rm = rm.reshape(shape)
    rv = rv.reshape(shape)
    y = (x - rm) / np.sqrt(rv + eps)
    if w is not None:
        y = y * w.reshape(shape)
    if b is not None:
        y = y + b.reshape(shape)
    return y


def conv2d(x: np.ndarray, weight: np.ndarray, bias: np.ndarray | None,
           stride=(1, 1), padding=(0, 0)) -> np.ndarray:
    """x: (C_in, H, W). weight: (C_out, C_in, KH, KW). Returns (C_out, H_out, W_out).
    Naive im2col-based gemm. No batch (B=1 implicit)."""
    Cin, H, W = x.shape
    Cout, _, KH, KW = weight.shape
    ph, pw = padding
    sh, sw = stride

    # Pad
    xp = np.pad(x, ((0, 0), (ph, ph), (pw, pw)))
    Hp, Wp = xp.shape[1], xp.shape[2]
    H_out = (Hp - KH) // sh + 1
    W_out = (Wp - KW) // sw + 1

    # im2col: (Cin*KH*KW, H_out*W_out)
    col = np.empty((Cin * KH * KW, H_out * W_out), dtype=np.float32)
    for kh in range(KH):
        for kw in range(KW):
            patch = xp[:, kh:kh + sh * H_out:sh, kw:kw + sw * W_out:sw]
            # patch shape: (Cin, H_out, W_out)
            col[(kh * KW + kw) * Cin:(kh * KW + kw + 1) * Cin, :] = patch.reshape(Cin, -1)
    # Weight reshape: (Cout, Cin*KH*KW)
    # Need to match the col layout: kh-major outer, then kw, then Cin.
    # Above col is (KH*KW*Cin, ...). Match by transposing weight:
    w_flat = weight.transpose(2, 3, 1, 0).reshape(KH * KW * Cin, Cout)  # (KH*KW*Cin, Cout)
    out = w_flat.T @ col   # (Cout, H_out*W_out)
    out = out.reshape(Cout, H_out, W_out).astype(np.float32)
    if bias is not None:
        out = out + bias.reshape(-1, 1, 1)
    return out


def conv1d(x: np.ndarray, weight: np.ndarray, bias: np.ndarray | None,
           stride: int = 1, padding: int = 0, dilation: int = 1) -> np.ndarray:
    """x: (C_in, T). weight: (C_out, C_in, K). Returns (C_out, T_out)."""
    Cin, T = x.shape
    Cout, _, K = weight.shape
    xp = np.pad(x, ((0, 0), (padding, padding)))
    Tp = xp.shape[1]
    eff_K = (K - 1) * dilation + 1
    T_out = (Tp - eff_K) // stride + 1

    # im2col
    col = np.empty((Cin * K, T_out), dtype=np.float32)
    for k in range(K):
        # dilated index
        start = k * dilation
        end = start + stride * T_out
        col[k * Cin:(k + 1) * Cin, :] = xp[:, start:end:stride]
    w_flat = weight.transpose(2, 1, 0).reshape(K * Cin, Cout)
    out = w_flat.T @ col
    out = out.reshape(Cout, T_out).astype(np.float32)
    if bias is not None:
        out = out + bias.reshape(-1, 1)
    return out


def seg_pooling(x: np.ndarray, seg_len: int = SEG_LEN) -> np.ndarray:
    """x: (C, T). avg_pool1d(kernel=seg_len, stride=seg_len, ceil_mode=True)
    then expand each segment back to seg_len so the result is (C, T) again
    (the upstream code crops back to the original T after expanding)."""
    C, T = x.shape
    n_segs = (T + seg_len - 1) // seg_len
    padded = n_segs * seg_len
    # The ceil-mode avg-pool with stride=kernel computes a mean PER FULL
    # SEGMENT; the trailing fractional segment (if any) averages only the
    # samples it contains. Implement as a per-segment sum / count.
    xp = np.zeros((C, padded), dtype=np.float32)
    xp[:, :T] = x
    valid = np.zeros((1, padded), dtype=np.float32)
    valid[:, :T] = 1.0
    sums   = xp.reshape(C, n_segs, seg_len).sum(axis=2)
    counts = valid.reshape(1, n_segs, seg_len).sum(axis=2).clip(min=1)
    pooled = sums / counts                                          # (C, n_segs)
    # Expand by repeating each segment seg_len times, then crop to T.
    expanded = np.repeat(pooled, seg_len, axis=1)[:, :T]
    return expanded.astype(np.float32)


# ---------------------------------------------------------------------------
# Layer-level helpers
# ---------------------------------------------------------------------------

def bn_relu_get(state: dict[str, np.ndarray], prefix: str,
                affine: bool = True) -> tuple:
    """Return (rm, rv, weight_or_None, bias_or_None) for a BatchNorm at `prefix`."""
    rm = state[prefix + ".running_mean"]
    rv = state[prefix + ".running_var"]
    w  = state[prefix + ".weight"] if affine else None
    b  = state[prefix + ".bias"]   if affine else None
    return rm, rv, w, b


def tdnn_layer(x: np.ndarray, state: dict, prefix: str,
               kernel_size: int, stride: int, padding: int,
               dilation: int = 1) -> np.ndarray:
    """Conv1d (no bias) -> BN -> ReLU."""
    w = state[prefix + ".linear.weight"]
    x = conv1d(x, w, None, stride=stride, padding=padding, dilation=dilation)
    rm, rv, gw, gb = bn_relu_get(state, prefix + ".nonlinear.batchnorm")
    x = bn(x, rm, rv, gw, gb)
    return relu(x)


def cam_layer(x: np.ndarray, state: dict, prefix: str,
              kernel_size: int, dilation: int) -> np.ndarray:
    """CAM (Context-Aware Masking) layer:
       y = linear_local(x);  context = mean(x) + seg_pool(x); m = sigmoid(linear2(relu(linear1(context))));  return y * m"""
    padding = (kernel_size - 1) // 2 * dilation
    w = state[prefix + ".linear_local.weight"]
    y = conv1d(x, w, None, stride=1, padding=padding, dilation=dilation)

    # Global + segment context
    ctx = x.mean(axis=-1, keepdims=True) + seg_pooling(x, SEG_LEN)
    # 1x1 conv to bn/reduction (reduction=2 in upstream)
    w1 = state[prefix + ".linear1.weight"]
    b1 = state[prefix + ".linear1.bias"]
    ctx = conv1d(ctx, w1, b1, stride=1, padding=0)
    ctx = relu(ctx)
    w2 = state[prefix + ".linear2.weight"]
    b2 = state[prefix + ".linear2.bias"]
    m = conv1d(ctx, w2, b2, stride=1, padding=0)
    m = sigmoid(m)
    return y * m


def cam_dense_tdnn_layer(x: np.ndarray, state: dict, prefix: str,
                          kernel_size: int, dilation: int) -> np.ndarray:
    """nonlinear1 (BN+ReLU) -> linear1 (1x1 conv, no bias) -> nonlinear2 (BN+ReLU) -> cam_layer."""
    rm, rv, gw, gb = bn_relu_get(state, prefix + ".nonlinear1.batchnorm")
    h = relu(bn(x, rm, rv, gw, gb))
    w = state[prefix + ".linear1.weight"]
    h = conv1d(h, w, None, stride=1, padding=0)              # bottleneck 1x1
    rm, rv, gw, gb = bn_relu_get(state, prefix + ".nonlinear2.batchnorm")
    h = relu(bn(h, rm, rv, gw, gb))
    h = cam_layer(h, state, prefix + ".cam_layer", kernel_size, dilation)
    return h


def cam_dense_tdnn_block(x: np.ndarray, state: dict, prefix: str,
                          num_layers: int, kernel_size: int,
                          dilation: int) -> np.ndarray:
    """DenseNet concat: each layer's output appended to channel dim."""
    for i in range(1, num_layers + 1):
        new = cam_dense_tdnn_layer(x, state, f"{prefix}.tdnnd{i}",
                                    kernel_size, dilation)
        x = np.concatenate([x, new], axis=0)
    return x


def transit_layer(x: np.ndarray, state: dict, prefix: str) -> np.ndarray:
    """BN+ReLU -> 1x1 conv. CAMPPlus passes bias=False, so no .linear.bias."""
    rm, rv, gw, gb = bn_relu_get(state, prefix + ".nonlinear.batchnorm")
    h = relu(bn(x, rm, rv, gw, gb))
    w = state[prefix + ".linear.weight"]
    b = state.get(prefix + ".linear.bias")        # None when bias=False
    return conv1d(h, w, b, stride=1, padding=0)


def dense_layer(x: np.ndarray, state: dict, prefix: str,
                affine: bool = True) -> np.ndarray:
    """1x1 conv -> BN+ReLU (or BN-only when affine=False)."""
    w = state[prefix + ".linear.weight"]
    # DenseLayer has bias=False by default in __init__
    h = conv1d(x, w, None, stride=1, padding=0)
    rm, rv, gw, gb = bn_relu_get(state, prefix + ".nonlinear.batchnorm", affine=affine)
    h = bn(h, rm, rv, gw, gb)
    if affine:
        h = relu(h)
    return h


# ---------------------------------------------------------------------------
# FCM (head)
# ---------------------------------------------------------------------------

def basic_res_block(x: np.ndarray, state: dict, prefix: str,
                     stride: int) -> np.ndarray:
    """Conv2d(s=(stride,1)) -> BN -> ReLU -> Conv2d(s=1) -> BN -> +shortcut -> ReLU.
       Shortcut adds a 1x1 conv + BN when stride != 1 or channels change."""
    w1 = state[prefix + ".conv1.weight"]
    out = conv2d(x, w1, None, stride=(stride, 1), padding=(1, 1))
    rm, rv, gw, gb = bn_relu_get(state, prefix + ".bn1")
    out = relu(bn(out, rm, rv, gw, gb))

    w2 = state[prefix + ".conv2.weight"]
    out = conv2d(out, w2, None, stride=(1, 1), padding=(1, 1))
    rm, rv, gw, gb = bn_relu_get(state, prefix + ".bn2")
    out = bn(out, rm, rv, gw, gb)

    # Shortcut: only present when stride != 1 or channels differ. In FCM
    # both layer1[0] and layer2[0] have stride=2 so both have shortcuts;
    # the other ResBlocks (stride=1, channels preserved) skip the shortcut.
    sc_w_key = prefix + ".shortcut.0.weight"
    if sc_w_key in state:
        sc_w = state[sc_w_key]
        sc = conv2d(x, sc_w, None, stride=(stride, 1), padding=(0, 0))
        rm, rv, gw, gb = bn_relu_get(state, prefix + ".shortcut.1")
        sc = bn(sc, rm, rv, gw, gb)
        out = out + sc
    else:
        out = out + x
    return relu(out)


def fcm(x: np.ndarray, state: dict) -> np.ndarray:
    """x: (80, T) fbank features. Returns (320, T)."""
    # Add channel dim: (1, 80, T)
    x = x[None, :, :]

    out = conv2d(x, state["head.conv1.weight"], None, stride=(1, 1), padding=(1, 1))
    rm, rv, gw, gb = bn_relu_get(state, "head.bn1")
    out = relu(bn(out, rm, rv, gw, gb))

    out = basic_res_block(out, state, "head.layer1.0", stride=2)
    out = basic_res_block(out, state, "head.layer1.1", stride=1)
    out = basic_res_block(out, state, "head.layer2.0", stride=2)
    out = basic_res_block(out, state, "head.layer2.1", stride=1)

    out = conv2d(out, state["head.conv2.weight"], None, stride=(2, 1), padding=(1, 1))
    rm, rv, gw, gb = bn_relu_get(state, "head.bn2")
    out = relu(bn(out, rm, rv, gw, gb))

    # Reshape: (C=32, F'=10, T) -> (320, T)
    C, F, T = out.shape
    return out.reshape(C * F, T).astype(np.float32)


# ---------------------------------------------------------------------------
# Full forward
# ---------------------------------------------------------------------------

def campplus_forward(state: dict, fbank: np.ndarray) -> np.ndarray:
    """fbank: (T, 80). Returns (192,) speaker embedding (NOT L2-normed)."""
    x = fbank.T.astype(np.float32)                                  # (80, T)

    x = fcm(x, state)                                                # (320, T)

    x = tdnn_layer(x, state, "xvector.tdnn",
                    kernel_size=5, stride=2, padding=2)              # (128, T/2-ish)

    for i, (n_layers, k, d) in enumerate(BLOCK_SPECS, start=1):
        x = cam_dense_tdnn_block(x, state, f"xvector.block{i}",
                                   num_layers=n_layers,
                                   kernel_size=k, dilation=d)
        x = transit_layer(x, state, f"xvector.transit{i}")

    rm, rv, gw, gb = bn_relu_get(state, "xvector.out_nonlinear.batchnorm")
    x = relu(bn(x, rm, rv, gw, gb))

    # StatsPool: mean + std over time
    mean = x.mean(axis=-1)
    std  = x.std (axis=-1, ddof=1)
    stats = np.concatenate([mean, std]).reshape(-1, 1)               # (2C, 1)

    # DenseLayer with config_str="batchnorm_" -> BN without affine.
    out = dense_layer(stats, state, "xvector.dense", affine=False)
    return out.reshape(-1).astype(np.float32)                         # (192,)


# ---------------------------------------------------------------------------
# Test input + driver
# ---------------------------------------------------------------------------

def make_test_fbank(T: int = 200, n_mels: int = FEAT_DIM) -> np.ndarray:
    """Seeded synthetic fbank. Range roughly matches Kaldi's
    log-fbank output (slightly-negative floats with O(unit) spread).
    Not real audio; this just exercises every code path."""
    rng = np.random.RandomState(19)
    base = rng.standard_normal((T, n_mels)).astype(np.float32) * 1.5
    # Bias so values are roughly in [-8, +2] like real log-fbank.
    base -= 3.0
    return base.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-bin",  default="tests/campplus_reference.bin")
    ap.add_argument("--out-meta", default="tests/campplus_reference.json")
    ap.add_argument("--frames",   type=int, default=200,
                    help="number of fbank frames in the test input")
    ap.add_argument("--dump-fcm-bin", default=None,
                    help="Also dump FCM head output to this path (debug bisect)")
    args = ap.parse_args()

    print(f"Fetching {S3GEN_FILE} ...")
    src = Path(hf_hub_download(REPO_ID, S3GEN_FILE))
    state_full = load_file(str(src))

    # Pull speaker_encoder.* tensors, rename to drop the prefix.
    state: dict[str, np.ndarray] = {}
    prefix = "speaker_encoder."
    for k, v in state_full.items():
        if k.startswith(prefix):
            state[k[len(prefix):]] = v.astype(np.float32)
    print(f"  speaker_encoder tensors: {len(state)}")

    fbank = make_test_fbank(T=args.frames)
    print(f"fbank input: {fbank.shape}")

    if args.dump_fcm_bin:
        # Just FCM, for debug bisect.
        x = fbank.T.astype(np.float32)        # (80, T)
        fcm_out = fcm(x, state)               # (320, T)
        print(f"FCM output: {fcm_out.shape}, min={fcm_out.min():+.4f} "
              f"max={fcm_out.max():+.4f} mean={fcm_out.mean():+.4f}")
        out = Path(args.dump_fcm_bin)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "wb") as f:
            # int32 channels, int32 T, then fp32 (channels, T) row-major
            f.write(np.int32(fcm_out.shape[0]).tobytes())
            f.write(np.int32(fcm_out.shape[1]).tobytes())
            f.write(fcm_out.tobytes())
        print(f"Wrote FCM dump to {out}")
        return

    emb = campplus_forward(state, fbank)
    print(f"embedding:   {emb.shape}")
    print(f"  min/max/avg: {float(emb.min()):+.4f} / {float(emb.max()):+.4f}"
          f" / {float(emb.mean()):+.4f}")
    print(f"  L2 norm:    {float(np.linalg.norm(emb)):.4f} (NOT normalized)")
    top5 = np.argsort(-np.abs(emb))[:5]
    print(f"  top-5 |dims|: {top5.tolist()}")
    print(f"  top-5 vals:   {emb[top5].tolist()}")

    out_bin = Path(args.out_bin)
    out_bin.parent.mkdir(parents=True, exist_ok=True)
    with open(out_bin, "wb") as f:
        f.write(np.int32(fbank.shape[0]).tobytes())   # T
        f.write(np.int32(fbank.shape[1]).tobytes())   # n_mels
        f.write(fbank.tobytes())                       # (T, n_mels)
        f.write(np.int32(emb.shape[0]).tobytes())      # emb_dim
        f.write(emb.tobytes())                          # (emb_dim,)

    Path(args.out_meta).write_text(json.dumps({
        "T":          int(fbank.shape[0]),
        "n_mels":     int(fbank.shape[1]),
        "emb_dim":    int(emb.shape[0]),
        "min":        float(emb.min()),
        "max":        float(emb.max()),
        "l2_norm":    float(np.linalg.norm(emb)),
        "top5_dims":  top5.tolist(),
        "top5_vals":  emb[top5].tolist(),
    }, indent=2), encoding="utf-8")

    print(f"\nWrote {out_bin} ({out_bin.stat().st_size} bytes)")
    print(f"Wrote {args.out_meta}")


if __name__ == "__main__":
    main()
