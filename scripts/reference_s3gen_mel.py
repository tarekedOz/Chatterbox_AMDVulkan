"""
NumPy reference for the S3Gen 24 kHz / 80-mel log spectrogram.

Mirrors upstream chatterbox/models/s3gen/utils/mel.py exactly. This is
the THIRD mel extractor in the project, distinct from both S3
Tokenizer's (128 mels @ 16 kHz, dynamic-range log) and VE's (40 mels
@ 16 kHz, log-amplitude).

Pipeline:
    audio (24 kHz mono fp32)
      -> reflect-pad by (n_fft - hop) / 2 = 720 samples each side
      -> STFT (n_fft=1920, hop=480, Hann window of size 1920, complex)
      -> magnitude = sqrt(real^2 + imag^2 + 1e-9)
      -> mel_filters @ magnitude     (librosa Slaney mel, n_mels=80,
                                       fmin=0, fmax=8000)
      -> log(clamp(min=1e-5))         (dynamic_range_compression_torch
                                       with C=1, clip_val=1e-5)

Output: tests/s3gen_mel_reference.bin
    int32   n_samples
    float32 * n_samples           waveform (24 kHz mono)
    int32   n_mels
    int32   n_frames
    float32 * (n_mels * n_frames) log-mel (n_mels, n_frames) row-major
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
import librosa

S3GEN_SR    = 24000
N_FFT       = 1920
HOP_SIZE    = 480
WIN_SIZE    = 1920
N_MELS      = 80
FMIN        = 0.0
FMAX        = 8000.0
LOG_CLIP    = 1e-5
MAG_EPS     = 1e-9


def make_test_wav(seconds: float = 1.0, sr: int = S3GEN_SR) -> np.ndarray:
    """Deterministic seeded mixed-tone waveform. Same family as the S3
    Tokenizer reference but at 24 kHz and with different frequencies
    (so the mel content differs)."""
    rng = np.random.RandomState(23)
    n = int(seconds * sr)
    t = np.arange(n, dtype=np.float32) / sr
    wav = (
        0.45 * np.sin(2 * np.pi * 261.0 * t)            # ~C4
        + 0.20 * np.sin(2 * np.pi * 698.0 * t + 0.3)    # ~F5, phase shift
        + 0.10 * np.sin(2 * np.pi * 1320.0 * t + 1.1)   # ~E6
        + 0.05 * rng.standard_normal(n).astype(np.float32)
    )
    wav *= 0.8 / max(np.max(np.abs(wav)), 1e-9)
    return wav.astype(np.float32)


def stft_centered(audio: np.ndarray, n_fft: int, hop: int,
                  window: np.ndarray) -> np.ndarray:
    """torch.stft(center=False)-equivalent on reflect-padded input.

    Upstream s3gen mel:
        audio_padded = F.pad(audio, ((n_fft - hop)//2, ...), mode='reflect')
        stft = torch.stft(audio_padded, n_fft, hop, win_size, window, center=False, ...)

    So we manually reflect-pad, then run STFT WITHOUT torch's auto-centering.
    Frame t covers padded[t*hop : t*hop + n_fft].

    Returns (n_fft//2 + 1, T) complex64.
    """
    pad = (n_fft - hop) // 2
    padded = np.pad(audio, pad, mode="reflect")
    n_frames = 1 + (len(padded) - n_fft) // hop
    out = np.empty((n_fft // 2 + 1, n_frames), dtype=np.complex64)
    for t in range(n_frames):
        frame = padded[t * hop : t * hop + n_fft] * window
        out[:, t] = np.fft.rfft(frame, n=n_fft).astype(np.complex64)
    return out


def log_mel_spectrogram(audio: np.ndarray,
                        mel_filters: np.ndarray,
                        window: np.ndarray) -> np.ndarray:
    """Match upstream s3gen utils/mel.py mel_spectrogram exactly."""
    stft = stft_centered(audio, N_FFT, HOP_SIZE, window)        # (961, T)
    mag = np.sqrt(stft.real ** 2 + stft.imag ** 2 + MAG_EPS).astype(np.float32)
    mel = (mel_filters @ mag).astype(np.float32)                  # (80, T)
    return np.log(np.clip(mel, LOG_CLIP, None)).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-bin",  default="tests/s3gen_mel_reference.bin")
    ap.add_argument("--out-meta", default="tests/s3gen_mel_reference.json")
    ap.add_argument("--out-filters-bin", default="tests/s3gen_mel_filters.bin",
                    help="Side-dump of the librosa-Slaney mel filterbank")
    ap.add_argument("--seconds", type=float, default=1.0)
    args = ap.parse_args()

    # librosa Slaney mel filterbank (the upstream default).
    mel_filters = librosa.filters.mel(
        sr=S3GEN_SR, n_fft=N_FFT, n_mels=N_MELS,
        fmin=FMIN, fmax=FMAX,
    ).astype(np.float32)
    print(f"mel filters: {mel_filters.shape}  "
          f"min={float(mel_filters.min()):+.5f}  "
          f"max={float(mel_filters.max()):+.5f}")

    window = np.hanning(WIN_SIZE).astype(np.float32)
    # np.hanning uses sym=True window in numpy by default; torch.hann_window
    # defaults to periodic=True which is sym=False. Need periodic Hann to
    # match torch.hann_window.
    # Standard periodic Hann of length N: w[n] = 0.5 - 0.5*cos(2*pi*n/N)
    # for n in 0..N-1 (NOT N-1 in denominator).
    n = np.arange(WIN_SIZE, dtype=np.float32)
    window = (0.5 - 0.5 * np.cos(2 * np.pi * n / WIN_SIZE)).astype(np.float32)

    wav = make_test_wav(seconds=args.seconds)
    print(f"WAV: {wav.shape[0]} samples @ {S3GEN_SR} Hz")

    log_mel = log_mel_spectrogram(wav, mel_filters, window)
    print(f"log-mel: {log_mel.shape}  "
          f"min={float(log_mel.min()):+.4f}  "
          f"max={float(log_mel.max()):+.4f}  "
          f"avg={float(log_mel.mean()):+.4f}")

    out_bin = Path(args.out_bin)
    out_bin.parent.mkdir(parents=True, exist_ok=True)
    n_mels, n_frames = log_mel.shape
    with open(out_bin, "wb") as f:
        f.write(np.int32(wav.shape[0]).tobytes())
        f.write(wav.tobytes())
        f.write(np.int32(n_mels).tobytes())
        f.write(np.int32(n_frames).tobytes())
        f.write(log_mel.tobytes())

    # Side-dump the mel filterbank for the C++ test's mel-filter-only
    # parity check. Format: int32 n_mels, int32 n_fft_bins, fp32[n_mels * n_fft_bins].
    fb = Path(args.out_filters_bin)
    with open(fb, "wb") as f:
        f.write(np.int32(mel_filters.shape[0]).tobytes())
        f.write(np.int32(mel_filters.shape[1]).tobytes())
        f.write(mel_filters.tobytes())

    Path(args.out_meta).write_text(json.dumps({
        "sample_rate": S3GEN_SR,
        "n_fft":       N_FFT,
        "hop_size":    HOP_SIZE,
        "win_size":    WIN_SIZE,
        "n_mels":      N_MELS,
        "fmin":        FMIN,
        "fmax":        FMAX,
        "n_samples":   int(wav.shape[0]),
        "n_frames":    int(n_frames),
        "mel_min":     float(log_mel.min()),
        "mel_max":     float(log_mel.max()),
        "mel_mean":    float(log_mel.mean()),
    }, indent=2), encoding="utf-8")

    print(f"\nWrote {out_bin} ({out_bin.stat().st_size} bytes)")
    print(f"Wrote {fb} ({fb.stat().st_size} bytes)")
    print(f"Wrote {args.out_meta}")


if __name__ == "__main__":
    main()
