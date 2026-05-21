"""
NumPy reference for the HiFTGenerator vocoder (s3gen/hifigan.py +
f0_predictor.py).

Architecture (Turbo config):

    mel (1, 80, T_mel) @ 50 Hz                        prompt 24 kHz mel
       |                                              from S3Gen prefix
       +--> F0 predictor (5x Conv1d-3 + ELU + Linear(512,1) + abs)
       |        -> f0 (1, T_mel)
       |        -> upsample by 480 (= 8*5*3*4) -> (1, T_mel*480, 1)
       |        -> SourceModuleHnNSF (SineGen with 8 harmonics + noise
       |                              + Linear(9,1) + tanh)
       |        -> source signal s (1, 1, T_mel*480)   (stochastic;
       |                                                we seed it and
       |                                                hand it over)
       |        -> STFT (n_fft=16, hop=4) -> (real(9, T_stft) +
       |                                       imag(9, T_stft))
       |        -> s_stft (1, 18, T_stft)
       |
       v
    conv_pre (Conv1d 80->512, k=7, p=3)  -> (1, 512, T_mel)
       |
       for i in [0..2]:
          x = leaky_relu(x, 0.1)
          x = ups[i](x)              ConvTranspose1d:
                                        i=0: 512->256, k=16, s=8, p=4
                                        i=1: 256->128, k=11, s=5, p=3
                                        i=2: 128->64,  k=7,  s=3, p=2
          if i == 2: x = ReflectionPad1d((1, 0))(x)
          si = source_downs[i](s_stft)        Conv1d:
                                        i=0: 18->256, k=30, s=15, p=7
                                        i=1: 18->128, k=6,  s=3,  p=1
                                        i=2: 18->64,  k=1,  s=1
          si = source_resblocks[i](si)
          x = x + si
          xs = sum(resblocks[i*3+j](x) for j in [0..2]) / 3
       end for
       |
       x = leaky_relu(x)
       x = conv_post (Conv1d 64->18, k=7, p=3)
       magnitude = exp(x[:, :9, :])
       phase     = sin(x[:, 9:, :])
       wav = iSTFT(magnitude, phase, n_fft=16, hop=4, window=hann(16))
       wav = clamp(wav, -0.99, 0.99)

ResBlock = stack of (Snake(C) -> Conv1d -> Snake(C) -> Conv1d) x 3
           with dilations [1, 3, 5] for the FIRST Conv1d and dilation=1
           for the SECOND. Residual added at each iteration.
           kernel_size: [3, 7, 11] for the main resblocks (3 per stage).
                        [7,  7, 11] for the source_resblocks (1 per stage).
           Snake: x + (1/alpha) * sin^2(x * alpha)
                  alpha is a learned (C,) parameter (linear scale).

The conv weights live under `parametrizations.weight.original{0,1}`:
  - original0: g (shape [1, 1, C_out]) -- magnitude
  - original1: v (shape [C_out, C_in, K]) -- direction
  - effective weight: w[c_out, ...] = g[c_out] * v[c_out, ...] / ||v[c_out, ...]||
  where the norm is over (C_in, K) (all dims except 0 in torch numpy).
This is torch's nn.utils.parametrizations.weight_norm. We
pre-combine at reference time.

Output binary: tests/hift_vocoder_reference.bin

  int32  T_mel
  int32  d_mel       = 80
  int32  n_fft       = 16
  int32  hop_len     = 4
  int32  upsample    = 480  (np.prod([8,5,3,4]))
  int32  T_wav       = T_mel * upsample
  int32  T_stft      = (T_mel*upsample - n_fft) // hop_len + 1 (centered STFT in torch default)
  int32  n_stages

  float32[T_mel * 80]            mel    (numpy (1, T_mel, 80), C inner)
  float32[T_mel]                  f0     (1, T_mel)
  float32[T_wav]                  source (1, 1, T_wav)
  float32[T_stft * 18]            s_stft (1, T_stft, 18)

  n_stages * (name + tensor)

  float32[T_wav]                  final_wav

The C++ test reads `source` directly as a precomputed input (since
SineGen uses two RNGs that are tedious to reproduce identically across
numpy and C++). f0 is also dumped so we can isolate the F0 predictor.
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

# Turbo HiFTGenerator config (from chatterbox/models/s3gen/s3gen.py defaults).
SAMPLING_RATE       = 24000
N_FFT               = 16
HOP_LEN             = 4
UPSAMPLE_RATES      = [8, 5, 3]
UPSAMPLE_KERNELS    = [16, 11, 7]
RESBLOCK_KERNELS    = [3, 7, 11]
RESBLOCK_DILATIONS  = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
SOURCE_RB_KERNELS   = [7, 7, 11]
SOURCE_RB_DILATIONS = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
NB_HARMONICS        = 8                 # 9 channels total (fundamental + 8)
NSF_ALPHA           = 0.1
NSF_SIGMA           = 0.003
NSF_VOICED_THR      = 10.0
LRELU_SLOPE         = 0.1
AUDIO_LIMIT         = 0.99
BASE_CHANNELS       = 512
IN_CHANNELS         = 80


# ---------------------------------------------------------------------------
# Activations / primitives
# ---------------------------------------------------------------------------

def elu(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    return np.where(x >= 0.0, x, alpha * (np.exp(x) - 1.0)).astype(np.float32)


def leaky_relu(x: np.ndarray, slope: float = 0.01) -> np.ndarray:
    return np.where(x >= 0.0, x, x * slope).astype(np.float32)


def snake(x: np.ndarray, alpha: np.ndarray,
            eps: float = 1e-9) -> np.ndarray:
    """Snake activation: x + 1/alpha * sin^2(x * alpha).
    alpha is (C,). x is (1, C, T). Broadcast alpha to (1, C, 1)."""
    a = alpha.reshape(1, -1, 1)
    return (x + (1.0 / (a + eps)) * np.sin(x * a) ** 2).astype(np.float32)


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


def conv_transpose1d(x: np.ndarray, w: np.ndarray, b: np.ndarray | None,
                       stride: int, padding: int) -> np.ndarray:
    """x: (C_in, T). w: (C_in, C_out, K). Returns (C_out, T_out).
       T_out = (T - 1) * stride + K - 2 * padding."""
    Cin, T = x.shape
    _, Cout, K = w.shape
    T_out_full = (T - 1) * stride + K
    out_full = np.zeros((Cout, T_out_full), dtype=np.float32)
    # For each input time step, add the kernel slid by stride to the output.
    for t in range(T):
        # x[:, t] has shape (Cin,). w has shape (Cin, Cout, K).
        # contribution at out[:, t*stride : t*stride + K] += w[Cin, Cout, K].T (summed over Cin)
        out_full[:, t * stride : t * stride + K] += np.einsum(
            "i,ijk->jk", x[:, t], w)
    # Strip padding on each side.
    out = out_full[:, padding : T_out_full - padding]
    if b is not None:
        out = out + b.reshape(-1, 1)
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Weight-norm pre-combine
# ---------------------------------------------------------------------------

def fuse_weight_norm(state: dict, prefix: str) -> tuple:
    """Look up parametrizations.weight.original0 (g) and .original1 (v)
    under `prefix`, return (w_fused, bias). Skips if not weight-normed
    (returns the plain .weight/.bias)."""
    g_key = prefix + ".parametrizations.weight.original0"
    v_key = prefix + ".parametrizations.weight.original1"
    b_key = prefix + ".bias"
    if g_key in state and v_key in state:
        g = state[g_key]               # (1, 1, C_out)
        v = state[v_key]               # (C_out, C_in, K)
        # Per torch's WeightNorm: dim=0 is the "norm" dim.
        # norm = sqrt(sum over all dims except 0)
        norm = np.sqrt((v ** 2).sum(axis=tuple(range(1, v.ndim)),
                                       keepdims=True))      # (C_out, 1, 1)
        w = g.reshape(-1, *([1] * (v.ndim - 1))) * v / norm
        b = state.get(b_key)
        return w.astype(np.float32), (b.astype(np.float32) if b is not None else None)
    # Plain weight.
    w = state[prefix + ".weight"]
    b = state.get(b_key)
    return w.astype(np.float32), (b.astype(np.float32) if b is not None else None)


# ---------------------------------------------------------------------------
# F0 predictor
# ---------------------------------------------------------------------------

def f0_predictor(mel: np.ndarray, state: dict) -> np.ndarray:
    """mel: (1, 80, T). Returns f0: (1, T)."""
    pfx = "mel2wav.f0_predictor"
    x = mel[0]            # (80, T)
    for i in [0, 2, 4, 6, 8]:
        w, b = fuse_weight_norm(state, f"{pfx}.condnet.{i}")
        x = conv1d(x, w, b, stride=1, padding=1, dilation=1)
        x = elu(x)
    # x: (512, T). Transpose to (T, 512), apply Linear(512, 1) + abs.
    x_T512 = x.T                                              # (T, 512)
    cw = state[f"{pfx}.classifier.weight"]                    # (1, 512)
    cb = state[f"{pfx}.classifier.bias"]                      # (1,)
    out = x_T512 @ cw.T + cb                                   # (T, 1)
    return np.abs(out[..., 0])[None, :].astype(np.float32)    # (1, T)


# ---------------------------------------------------------------------------
# F0 upsample + NSF source (SineGen + SourceModuleHnNSF)
# ---------------------------------------------------------------------------

def nearest_upsample(x: np.ndarray, factor: int) -> np.ndarray:
    """x: (..., T) -> (..., T*factor) with nearest-neighbor repeat."""
    return np.repeat(x, factor, axis=-1)


def nsf_source(f0_up: np.ndarray, state: dict,
                seed: int = 7, return_rng_inputs: bool = False
                 ) -> "np.ndarray | tuple":
    """f0_up: (1, T_wav). Returns s: (1, 1, T_wav).
    If return_rng_inputs is True, also returns the (phase_vec, noise)
    random arrays so the C++ side can consume the same values."""
    rng = np.random.RandomState(seed)
    B, T = f0_up.shape
    # SineGen input: (B, 1, T)
    f0_BCT = f0_up.reshape(B, 1, T)

    # F_mat[i] = f0 * (i+1) / sample_rate
    F_mat = np.zeros((B, NB_HARMONICS + 1, T), dtype=np.float32)
    for i in range(NB_HARMONICS + 1):
        F_mat[:, i:i+1, :] = f0_BCT * (i + 1) / SAMPLING_RATE
    # theta = 2*pi * (cumsum(F) % 1)
    cs = np.cumsum(F_mat, axis=-1)
    theta = (2.0 * np.pi * (cs - np.floor(cs))).astype(np.float32)
    # Random phase per (B, harmonic) — but first harmonic phase = 0.
    phase = rng.uniform(-math.pi, math.pi,
                          size=(B, NB_HARMONICS + 1, 1)).astype(np.float32)
    phase[:, 0, :] = 0.0
    sine = NSF_ALPHA * np.sin(theta + phase).astype(np.float32)
    # U/V mask (B, 1, T)
    uv = (f0_BCT > NSF_VOICED_THR).astype(np.float32)
    # Noise
    noise_amp = uv * NSF_SIGMA + (1.0 - uv) * NSF_ALPHA / 3.0
    noise_z = rng.standard_normal(sine.shape).astype(np.float32)
    noise = (noise_amp * noise_z).astype(np.float32)
    sine_waves = (sine * uv + noise).astype(np.float32)        # (B, 9, T)
    # SourceModuleHnNSF: Linear(9, 1) + tanh, applied on (B, T, 9) input.
    sw = sine_waves.transpose(0, 2, 1)                          # (B, T, 9)
    lw = state["mel2wav.m_source.l_linear.weight"]              # (1, 9)
    lb = state["mel2wav.m_source.l_linear.bias"]                # (1,)
    merged = sw @ lw.T + lb                                      # (B, T, 1)
    s = np.tanh(merged).astype(np.float32)                      # (B, T, 1)
    s = s.transpose(0, 2, 1).astype(np.float32)                 # (B, 1, T)
    if return_rng_inputs:
        return s, phase.astype(np.float32), noise_z.astype(np.float32)
    return s


# ---------------------------------------------------------------------------
# STFT / iSTFT
# ---------------------------------------------------------------------------

def hann_window(N: int, periodic: bool = True) -> np.ndarray:
    if periodic:
        n = np.arange(N, dtype=np.float32)
        return (0.5 - 0.5 * np.cos(2.0 * np.pi * n / N)).astype(np.float32)
    # symmetric Hann (window of size N-1, then padded)
    n = np.arange(N, dtype=np.float32)
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * n / (N - 1))).astype(np.float32)


def stft_centered(audio: np.ndarray, n_fft: int, hop: int,
                    window: np.ndarray) -> tuple:
    """torch.stft default (center=True, pad_mode='reflect', return_complex=True).
    Reflects-pad by n_fft//2 each side, then STFT.
    audio: (B, T). Returns (B, n_fft//2+1, T_stft) real and imag."""
    B, T = audio.shape
    pad = n_fft // 2
    padded = np.pad(audio, ((0, 0), (pad, pad)), mode="reflect")
    T_stft = 1 + (padded.shape[1] - n_fft) // hop
    n_bins = n_fft // 2 + 1
    real = np.zeros((B, n_bins, T_stft), dtype=np.float32)
    imag = np.zeros((B, n_bins, T_stft), dtype=np.float32)
    for b in range(B):
        for t in range(T_stft):
            frame = padded[b, t * hop : t * hop + n_fft] * window
            spec = np.fft.rfft(frame, n=n_fft)
            real[b, :, t] = spec.real
            imag[b, :, t] = spec.imag
    return real.astype(np.float32), imag.astype(np.float32)


def istft_centered(real: np.ndarray, imag: np.ndarray,
                     n_fft: int, hop: int, window: np.ndarray) -> np.ndarray:
    """torch.istft inverse of stft_centered. real/imag: (B, n_fft//2+1, T_stft).
    Returns audio: (B, T_audio = (T_stft - 1) * hop)."""
    B, n_bins, T_stft = real.shape
    assert n_bins == n_fft // 2 + 1

    # Overlap-add with windowing.
    audio_len_padded = (T_stft - 1) * hop + n_fft
    audio_pad = np.zeros((B, audio_len_padded), dtype=np.float32)
    weight   = np.zeros((B, audio_len_padded), dtype=np.float32)
    for b in range(B):
        for t in range(T_stft):
            spec = real[b, :, t] + 1j * imag[b, :, t]
            frame = np.fft.irfft(spec, n=n_fft).astype(np.float32) * window
            audio_pad[b, t * hop : t * hop + n_fft] += frame
            weight  [b, t * hop : t * hop + n_fft] += window ** 2
    audio_pad = audio_pad / np.where(weight > 1e-11, weight, 1.0)
    # Strip the n_fft//2 padding on each side.
    pad = n_fft // 2
    return audio_pad[:, pad : audio_len_padded - pad].astype(np.float32)


# ---------------------------------------------------------------------------
# ResBlock
# ---------------------------------------------------------------------------

def res_block(x_CT: np.ndarray, state: dict, prefix: str,
                kernel_size: int, dilations: list,
                channels: int) -> np.ndarray:
    """x_CT: (C, T). 3 (Snake -> Conv -> Snake -> Conv) blocks with residual."""
    pad = (kernel_size - 1) // 2 * 1   # for d=1 paddings
    for i, d in enumerate(dilations):
        # alpha for activations1[i] and activations2[i]
        a1 = state[f"{prefix}.activations1.{i}.alpha"]
        a2 = state[f"{prefix}.activations2.{i}.alpha"]
        # Snake1
        xt = snake(x_CT[None], a1)[0]                            # (C, T)
        # Conv1 (dilation=d)
        w1, b1 = fuse_weight_norm(state, f"{prefix}.convs1.{i}")
        p_d = (kernel_size - 1) // 2 * d
        xt = conv1d(xt, w1, b1, stride=1, padding=p_d, dilation=d)
        # Snake2
        xt = snake(xt[None], a2)[0]
        # Conv2 (dilation=1)
        w2, b2 = fuse_weight_norm(state, f"{prefix}.convs2.{i}")
        p_1 = (kernel_size - 1) // 2
        xt = conv1d(xt, w2, b2, stride=1, padding=p_1, dilation=1)
        x_CT = (xt + x_CT).astype(np.float32)
    (void := channels)
    return x_CT


def reflection_pad1d(x_CT: np.ndarray, pad_left: int,
                       pad_right: int) -> np.ndarray:
    return np.pad(x_CT, ((0, 0), (pad_left, pad_right)), mode="reflect")


# ---------------------------------------------------------------------------
# Full decode forward
# ---------------------------------------------------------------------------

def decode(mel: np.ndarray, source: np.ndarray, state: dict,
             dumps: dict) -> np.ndarray:
    """mel: (1, 80, T_mel). source: (1, 1, T_wav). Returns wav: (1, T_wav)."""
    B, _, T_mel = mel.shape
    T_wav = source.shape[-1]

    # STFT of the source signal.
    window = hann_window(N_FFT, periodic=True)
    s_real, s_imag = stft_centered(source[:, 0], N_FFT, HOP_LEN, window)
    s_stft = np.concatenate([s_real, s_imag], axis=1)   # (B, 18, T_stft)
    dumps["source_stft"] = s_stft.copy()
    T_stft = s_stft.shape[-1]

    # conv_pre
    cw, cb = fuse_weight_norm(state, "mel2wav.conv_pre")
    x = conv1d(mel[0], cw, cb, stride=1, padding=3)                # (512, T_mel)
    dumps["after_conv_pre"] = x[None].astype(np.float32)            # (1, 512, T_mel)

    # 3 upsample stages
    for i in range(3):
        x = leaky_relu(x, LRELU_SLOPE)
        # ups[i]: ConvTranspose1d(C, C/2, k=UPSAMPLE_KERNELS[i], s=UPSAMPLE_RATES[i],
        #                        pad=(k-s)//2)
        uw, ub = fuse_weight_norm(state, f"mel2wav.ups.{i}")
        # original1 shape in numpy is (C_in, C_out, K). ConvTranspose1d expects that order.
        s = UPSAMPLE_RATES[i]
        k = UPSAMPLE_KERNELS[i]
        p = (k - s) // 2
        x = conv_transpose1d(x, uw, ub, stride=s, padding=p)         # (C/2, T*s)
        dumps[f"after_ups{i}"] = x[None].astype(np.float32)

        if i == 2:
            x = reflection_pad1d(x, 1, 0)
            dumps["after_refl_pad"] = x[None].astype(np.float32)

        # source_downs[i]
        sd_w, sd_b = fuse_weight_norm(state, f"mel2wav.source_downs.{i}")
        # Use upstream stride/padding rules.
        cum_rates = [15, 3, 1]                # downsample_cum_rates reversed
        u = cum_rates[i]
        if u == 1:
            si = conv1d(s_stft[0], sd_w, sd_b, stride=1, padding=0)
        else:
            si = conv1d(s_stft[0], sd_w, sd_b, stride=u, padding=u // 2)
        # source_resblocks[i]
        ch = BASE_CHANNELS // (2 ** (i + 1))
        si = res_block(si, state, f"mel2wav.source_resblocks.{i}",
                          SOURCE_RB_KERNELS[i], SOURCE_RB_DILATIONS[i],
                          ch)
        dumps[f"after_src_rb{i}"] = si[None].astype(np.float32)

        # Add source to x
        x = (x + si).astype(np.float32)
        dumps[f"after_src_add{i}"] = x[None].astype(np.float32)

        # Parallel ResBlock kernels — sum, then divide.
        xs = None
        for j in range(3):
            r = res_block(x, state, f"mel2wav.resblocks.{i * 3 + j}",
                             RESBLOCK_KERNELS[j], RESBLOCK_DILATIONS[j],
                             ch)
            xs = r if xs is None else (xs + r)
        x = (xs / 3.0).astype(np.float32)
        dumps[f"after_rb_avg{i}"] = x[None].astype(np.float32)

    # conv_post
    x = leaky_relu(x)
    cw, cb = fuse_weight_norm(state, "mel2wav.conv_post")
    x = conv1d(x, cw, cb, stride=1, padding=3)                       # (18, T_final)
    dumps["after_conv_post"] = x[None].astype(np.float32)
    n_bins = N_FFT // 2 + 1
    magnitude = np.exp(x[:n_bins])                                    # (9, T_final)
    phase     = np.sin(x[n_bins:])                                    # (9, T_final)
    magnitude = np.clip(magnitude, None, 1e2)
    dumps["magnitude"] = magnitude[None].astype(np.float32)
    dumps["phase"]     = phase[None].astype(np.float32)
    real = magnitude * np.cos(phase)
    imag = magnitude * np.sin(phase)
    wav = istft_centered(real[None], imag[None], N_FFT, HOP_LEN, window)
    wav = np.clip(wav, -AUDIO_LIMIT, AUDIO_LIMIT)
    return wav.astype(np.float32)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def make_test_mel(T_mel: int, seed: int = 53) -> np.ndarray:
    """Seeded mel-like input. Values roughly resembling a log-mel
    spectrogram (negative most of the time, mildly varying)."""
    rng = np.random.RandomState(seed)
    mel = rng.standard_normal((1, IN_CHANNELS, T_mel)).astype(np.float32) * 0.5
    mel -= 2.0
    return mel.astype(np.float32)


def write_stage(f, name: str, t: np.ndarray) -> None:
    """Dump (1, T, C) layout so bytes match ggml ne=(C, T)."""
    nb = name.encode("ascii")
    assert len(nb) <= 31
    f.write(nb + b"\x00" * (32 - len(nb)))
    if t.ndim == 2:
        t = t[None, :, :]
    assert t.ndim == 3, f"{name}: expected 3D, got {t.shape}"
    if t.shape[1] != 1 and t.shape[2] != 1:
        t = t.transpose(0, 2, 1)
    d0, d1, d2 = t.shape
    f.write(np.int32(d0).tobytes())
    f.write(np.int32(d1).tobytes())
    f.write(np.int32(d2).tobytes())
    f.write(np.ascontiguousarray(t.astype(np.float32)).tobytes())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-bin",  default="tests/hift_vocoder_reference.bin")
    ap.add_argument("--out-meta", default="tests/hift_vocoder_reference.json")
    ap.add_argument("--T-mel",    type=int, default=16,
                    help="Number of mel frames at 50 Hz (default 16 -> ~320 ms @ 24 kHz).")
    args = ap.parse_args()

    print(f"Fetching {S3GEN_FILE} ...")
    src = Path(hf_hub_download(REPO_ID, S3GEN_FILE))
    state_full = load_file(str(src))
    state = {k: v.astype(np.float32) for k, v in state_full.items()
              if k.startswith("mel2wav.")}
    print(f"  mel2wav.* tensors: {len(state)}")

    T_mel = args.T_mel
    mel = make_test_mel(T_mel)
    print(f"mel: {mel.shape}  min={mel.min():.3f} max={mel.max():.3f}")

    # F0 prediction.
    f0 = f0_predictor(mel, state)                       # (1, T_mel)
    print(f"f0 (1, T_mel): min={f0.min():.3f} max={f0.max():.3f} mean={f0.mean():.3f}")

    # F0 upsampling (nearest 480x).
    upsample_factor = int(np.prod(UPSAMPLE_RATES)) * HOP_LEN
    f0_up = nearest_upsample(f0, upsample_factor)        # (1, T_wav)
    T_wav = f0_up.shape[-1]
    print(f"f0_up: T_wav={T_wav}")

    # NSF source (seeded for determinism). Also dump the RNG inputs so
    # the C++ NSF generator can consume them and produce identical output.
    source, phase_vec, noise_z = nsf_source(f0_up, state, seed=7,
                                              return_rng_inputs=True)
    print(f"source: {source.shape}  min={source.min():.4f} max={source.max():.4f}")
    print(f"phase_vec: {phase_vec.shape}  noise_z: {noise_z.shape}")

    # Decode.
    dumps: dict = {}
    wav = decode(mel, source, state, dumps)
    wav = wav.astype(np.float32)
    print(f"\nwav: shape={wav.shape}  min={wav.min():.4f} max={wav.max():.4f}  "
          f"mean={wav.mean():.4f}")
    for k, t in dumps.items():
        print(f"  {k:24s} shape={t.shape}  min={t.min():+.4f} max={t.max():+.4f}")

    # Write binary.
    out_bin = Path(args.out_bin)
    out_bin.parent.mkdir(parents=True, exist_ok=True)
    with open(out_bin, "wb") as f:
        f.write(np.int32(T_mel).tobytes())
        f.write(np.int32(IN_CHANNELS).tobytes())
        f.write(np.int32(N_FFT).tobytes())
        f.write(np.int32(HOP_LEN).tobytes())
        f.write(np.int32(upsample_factor).tobytes())
        f.write(np.int32(T_wav).tobytes())
        T_stft = dumps["source_stft"].shape[-1]
        f.write(np.int32(T_stft).tobytes())
        f.write(np.int32(len(dumps)).tobytes())

        # mel (1, T, 80)
        mel_TC = np.ascontiguousarray(mel[0].T.astype(np.float32))
        f.write(mel_TC.tobytes())
        # f0 (1, T)
        f.write(f0[0].astype(np.float32).tobytes())
        # source (1, T_wav) — 1D
        f.write(source[0, 0].astype(np.float32).tobytes())
        # NSF RNG inputs (so the C++ side can match exactly):
        #   phase_vec (NB_HARMONICS + 1,) — 9 floats; index 0 is always 0
        f.write(phase_vec[0, :, 0].astype(np.float32).tobytes())
        #   noise_z   ((NB_HARMONICS + 1) * T_wav,) row-major (harmonic, time)
        f.write(np.ascontiguousarray(noise_z[0].astype(np.float32)).tobytes())
        # s_stft already (1, 18, T_stft) -> (T_stft, 18)
        s_stft = dumps["source_stft"]
        s_stft_TC = np.ascontiguousarray(s_stft[0].T.astype(np.float32))
        f.write(s_stft_TC.tobytes())

        for name, t in dumps.items():
            write_stage(f, name, t)

        # Final wav (1, T_wav) - 1D
        f.write(wav[0].astype(np.float32).tobytes())

    Path(args.out_meta).write_text(json.dumps({
        "T_mel":     int(T_mel),
        "T_wav":     int(T_wav),
        "T_stft":    int(T_stft),
        "n_fft":     N_FFT,
        "hop_len":   HOP_LEN,
        "n_stages":  len(dumps),
        "stages":    list(dumps.keys()),
        "f0_min":    float(f0.min()),
        "f0_max":    float(f0.max()),
        "f0_mean":   float(f0.mean()),
        "wav_min":   float(wav.min()),
        "wav_max":   float(wav.max()),
        "wav_mean":  float(wav.mean()),
    }, indent=2), encoding="utf-8")
    print(f"\nWrote {out_bin} ({out_bin.stat().st_size} bytes)")
    print(f"Wrote {args.out_meta}")


if __name__ == "__main__":
    main()
