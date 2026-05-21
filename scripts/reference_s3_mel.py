"""
NumPy reference for the S3 tokenizer's log-mel spectrogram.

Mirrors S3Tokenizer.log_mel_spectrogram in upstream chatterbox:
    stft(audio, n_fft=400, hop=160, window=hann_400, return_complex=True)
    mag = stft[..., :-1].abs() ** 2
    mel = mel_filters @ mag
    log = clamp(mel, min=1e-10).log10()
    log = max(log, log.max() - 8.0)
    log = (log + 4) / 4

Loads the GGUF-baked filterbank + window so the result is identical to
running upstream against the same WAV. Uses a deterministic synthetic
1-second waveform for testing (no real audio file needed).

Output: tests/s3_mel_reference.bin (binary)
    int32   n_samples
    float32 * n_samples         waveform (16 kHz mono)
    int32   n_mels
    int32   n_frames
    float32 * (n_mels * n_frames)   log-mel (row-major mel[m, t])
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
S3GEN_FILE = "s3gen_meanflow.safetensors"

S3_SR  = 16000
N_FFT  = 400
N_HOP  = 160
N_MELS = 128
N_FFT_BINS = N_FFT // 2 + 1   # 201


def make_test_wav(seconds: float = 1.0, sr: int = S3_SR) -> np.ndarray:
    """A deterministic mix of two sines + low-amplitude noise. Not real
    speech, but enough harmonic content to exercise every mel bin."""
    rng = np.random.RandomState(13)
    n = int(seconds * sr)
    t = np.arange(n, dtype=np.float32) / sr
    wav = (
        0.40 * np.sin(2 * np.pi * 220.0 * t)        # A3
        + 0.20 * np.sin(2 * np.pi * 880.0 * t + 0.7)  # A5, phase shift
        + 0.05 * rng.standard_normal(n).astype(np.float32)
    )
    # Normalize to peak ~0.8 to avoid clipping concerns.
    wav *= 0.8 / max(np.max(np.abs(wav)), 1e-9)
    return wav.astype(np.float32)


def stft_centered(audio: np.ndarray, n_fft: int, hop: int,
                  window: np.ndarray) -> np.ndarray:
    """torch.stft-equivalent with center=True. Reflect-pads by n_fft//2
    on each side, then strides n_fft-wide windows by `hop`.

    Returns (n_fft//2 + 1, T) complex64.
    """
    pad = n_fft // 2
    padded = np.pad(audio, pad, mode="reflect")
    n_frames = 1 + (len(padded) - n_fft) // hop
    out = np.empty((n_fft // 2 + 1, n_frames), dtype=np.complex64)
    for t in range(n_frames):
        start = t * hop
        frame = padded[start:start + n_fft] * window
        spec = np.fft.rfft(frame, n=n_fft)
        out[:, t] = spec.astype(np.complex64)
    return out


def log_mel_spectrogram(audio: np.ndarray,
                        mel_filters: np.ndarray,
                        window: np.ndarray) -> np.ndarray:
    """Match S3Tokenizer.log_mel_spectrogram exactly."""
    stft = stft_centered(audio, N_FFT, N_HOP, window)        # (201, T+1) typically
    mag  = (np.abs(stft[:, :-1]) ** 2).astype(np.float32)     # drop last time frame
    mel  = mel_filters @ mag                                  # (128, T)
    log  = np.log10(np.clip(mel, 1e-10, None))
    log  = np.maximum(log, log.max() - 8.0)
    log  = (log + 4.0) / 4.0
    return log.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-bin",  default="tests/s3_mel_reference.bin")
    ap.add_argument("--out-meta", default="tests/s3_mel_reference.json")
    ap.add_argument("--seconds",  type=float, default=1.0)
    args = ap.parse_args()

    print(f"Fetching {S3GEN_FILE} ...")
    src = Path(hf_hub_download(REPO_ID, S3GEN_FILE))
    state = load_file(str(src))

    mel_filters = state["tokenizer._mel_filters"].astype(np.float32)
    window      = state["tokenizer.window"].astype(np.float32)
    print(f"  mel_filters: {mel_filters.shape}  window: {window.shape}")
    assert mel_filters.shape == (N_MELS, N_FFT_BINS)
    assert window.shape == (N_FFT,)

    wav = make_test_wav(seconds=args.seconds)
    print(f"Test WAV: {wav.shape[0]} samples @ {S3_SR} Hz ({wav.shape[0]/S3_SR:.2f} s)")

    log_mel = log_mel_spectrogram(wav, mel_filters, window)
    print(f"log-mel shape:   {log_mel.shape}")
    print(f"log-mel min/max: {float(log_mel.min()):+.4f} / {float(log_mel.max()):+.4f}")
    print(f"log-mel mean:    {float(log_mel.mean()):+.4f}")

    out_bin = Path(args.out_bin)
    out_bin.parent.mkdir(parents=True, exist_ok=True)
    n_mels, n_frames = log_mel.shape
    with open(out_bin, "wb") as f:
        f.write(np.int32(wav.shape[0]).tobytes())
        f.write(wav.tobytes())
        f.write(np.int32(n_mels).tobytes())
        f.write(np.int32(n_frames).tobytes())
        f.write(log_mel.tobytes())

    Path(args.out_meta).write_text(json.dumps({
        "sample_rate": S3_SR,
        "n_fft": N_FFT,
        "hop_length": N_HOP,
        "n_mels": int(n_mels),
        "n_frames": int(n_frames),
        "n_samples": int(wav.shape[0]),
        "mel_min": float(log_mel.min()),
        "mel_max": float(log_mel.max()),
        "mel_mean": float(log_mel.mean()),
    }, indent=2), encoding="utf-8")

    print(f"\nWrote {out_bin} ({out_bin.stat().st_size} bytes)")
    print(f"Wrote {args.out_meta}")


if __name__ == "__main__":
    main()
