"""
NumPy reference for the C++ VE 40-mel @ 16 kHz extractor.

This is the FOURTH mel-family extractor in the project, used as input
to the VoiceEncoder LSTM (T3's speaker conditioning):

  S3Tokenizer mel:  128 mels @ 16 kHz, n_fft=400, hop=160, power=2,
                    no slaney norm, dynamic-range-clipped log.
  S3Gen mel:         80 mels @ 24 kHz, n_fft=1920, hop=480, sqrt-magnitude,
                    slaney mel + slaney norm, log+1e-5.
  Kaldi fbank:       80 mels @ 16 kHz, frame_len=400, frame_shift=160,
                    HTK mel, povey window, power, preemph 0.97.
  VE mel (THIS):     40 mels @ 16 kHz, n_fft=400, hop=160, power=2,
                    slaney mel + slaney norm, log+1e-5.

Mirrors chatterbox/models/voice_encoder/melspec.py exactly.

Output: tests/ve_mel_reference.bin
    int32  n_samples
    int32  n_mels
    int32  n_frames
    float32[n_samples]           audio
    float32[n_frames * n_mels]   mel (row-major (T, 40))
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


SR        = 16000
N_FFT     = 400
HOP       = 160
WIN       = 400
N_MELS    = 40
FMIN      = 0.0
FMAX      = 8000.0     # sr/2
LOG_CLIP  = 1e-5


def make_test_audio(seconds: float = 1.0, seed: int = 31) -> np.ndarray:
    rng = np.random.RandomState(seed)
    t = np.arange(int(seconds * SR), dtype=np.float32) / SR
    audio = (
        0.40 * np.sin(2 * np.pi * 200.0 * t)
        + 0.25 * np.sin(2 * np.pi * 720.0 * t + 0.4)
        + 0.10 * np.sin(2 * np.pi * 1850.0 * t + 1.1)
        + 0.04 * rng.standard_normal(t.shape[0]).astype(np.float32)
    )
    audio *= 0.5 / max(np.max(np.abs(audio)), 1e-9)
    return audio.astype(np.float32)


def compute_ve_mel(audio: np.ndarray) -> np.ndarray:
    """Match librosa.feature.melspectrogram(audio, sr=16000, n_fft=400,
    hop_length=160, n_mels=40, power=2.0) + log+1e-5 floor."""
    mel_filters = librosa.filters.mel(
        sr=SR, n_fft=N_FFT, n_mels=N_MELS, fmin=FMIN, fmax=FMAX,
    ).astype(np.float32)
    # Use librosa STFT (centered, reflect-pad).
    stft = librosa.stft(audio, n_fft=N_FFT, hop_length=HOP, win_length=WIN,
                          window="hann", center=True, pad_mode="reflect")
    power = (stft.real ** 2 + stft.imag ** 2).astype(np.float32)   # power spectrum
    mel = (mel_filters @ power).astype(np.float32)                  # (40, T)
    log_mel = np.log(np.clip(mel, LOG_CLIP, None)).astype(np.float32)
    return log_mel.T.astype(np.float32)        # (T, 40)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-bin",  default="tests/ve_mel_reference.bin")
    ap.add_argument("--out-meta", default="tests/ve_mel_reference.json")
    ap.add_argument("--seconds",  type=float, default=1.0)
    args = ap.parse_args()

    audio = make_test_audio(args.seconds)
    mel = compute_ve_mel(audio)
    print(f"audio: {audio.shape}  min={audio.min():.4f} max={audio.max():.4f}")
    print(f"mel:   shape={mel.shape}  min={mel.min():.4f} max={mel.max():.4f} "
          f"mean={mel.mean():.4f}")

    out = Path(args.out_bin)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        f.write(np.int32(audio.shape[0]).tobytes())
        f.write(np.int32(N_MELS).tobytes())
        f.write(np.int32(mel.shape[0]).tobytes())
        f.write(audio.tobytes())
        f.write(np.ascontiguousarray(mel.astype(np.float32)).tobytes())

    Path(args.out_meta).write_text(json.dumps({
        "sr":       SR,
        "n_samples": int(audio.shape[0]),
        "n_mels":   N_MELS,
        "n_frames": int(mel.shape[0]),
        "mel_min":  float(mel.min()),
        "mel_max":  float(mel.max()),
        "mel_mean": float(mel.mean()),
    }, indent=2), encoding="utf-8")
    print(f"\nWrote {out}  ({out.stat().st_size} bytes)")
    print(f"Wrote {args.out_meta}")


if __name__ == "__main__":
    main()
