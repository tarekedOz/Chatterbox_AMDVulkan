"""
NumPy reference for the C++ Kaldi-style 80-d fbank extractor.

Mirrors torchaudio.compliance.kaldi.fbank exactly with the parameters
the chatterbox CAMPPlus path uses:

  num_mel_bins=80, sample_frequency=16000, dither=0.0
  (other params: kaldi defaults — frame_length=25ms, frame_shift=10ms,
   preemphasis=0.97, mel_scale='htk', window_type='povey',
   use_energy=False, use_power=True, use_log_fbank=True,
   low_freq=20.0, high_freq=0 (=sample_freq/2), subtract_mean=False,
   remove_dc_offset=True, raw_energy=True, round_to_power_of_two=True)

Algorithm (per torchaudio source):
  For each frame n*hop_shift .. n*hop_shift + window_length:
    1. Slice frame from waveform (windowed by frame_length samples).
    2. dither (skip — dither=0).
    3. remove_dc_offset: x -= mean(x)
    4. preemphasis: y[0] = x[0] - 0.97*x[0]; y[k] = x[k] - 0.97*x[k-1]
    5. window with povey window of length frame_length:
       w[k] = (0.5 - 0.5 cos(2π k / (frame_length - 1))) ** 0.85
    6. zero-pad to FFT size (next power of 2 >= frame_length = 512)
    7. real FFT -> magnitude^2 (power spectrum)
    8. mel filterbank @ HTK scale, num_mel_bins=80:
         mel(f) = 1127 * ln(1 + f/700)        (HTK mel — natural log)
       Wait: torchaudio uses 1127.01048 base. Verify in code.
    9. log(max(mel_energy, EPSILON))    where EPSILON = 1.1754944e-38

Output binary: tests/kaldi_fbank_reference.bin

    int32   sample_rate
    int32   n_samples
    int32   num_mel_bins
    int32   n_frames
    float32[n_samples]                       audio
    float32[n_frames * num_mel_bins]         expected_fbank (T, n_mels)
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
import torch
import torchaudio.compliance.kaldi as kaldi


def make_test_signal(sr: int = 16000, seconds: float = 1.0,
                       seed: int = 29) -> np.ndarray:
    rng = np.random.RandomState(seed)
    t = np.arange(int(sr * seconds), dtype=np.float32) / sr
    # Three sines + light noise, normalized to ~0.5 max amplitude.
    audio = (
        0.30 * np.sin(2 * np.pi * 180.0 * t)
        + 0.20 * np.sin(2 * np.pi * 420.0 * t + 0.6)
        + 0.10 * np.sin(2 * np.pi * 1700.0 * t + 1.2)
        + 0.03 * rng.standard_normal(t.shape[0]).astype(np.float32)
    )
    audio *= 0.5 / max(np.max(np.abs(audio)), 1e-9)
    return audio.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-bin",  default="tests/kaldi_fbank_reference.bin")
    ap.add_argument("--out-meta", default="tests/kaldi_fbank_reference.json")
    ap.add_argument("--seconds",  type=float, default=1.0)
    args = ap.parse_args()

    SR        = 16000
    N_MELS    = 80
    audio = make_test_signal(SR, args.seconds)
    print(f"audio: {audio.shape}  min={audio.min():.4f} max={audio.max():.4f}")

    fb = kaldi.fbank(
        torch.from_numpy(audio).unsqueeze(0),
        num_mel_bins=N_MELS,
        sample_frequency=SR,
        dither=0.0,
    ).numpy().astype(np.float32)
    print(f"fbank: shape={fb.shape}  min={fb.min():.4f}  max={fb.max():.4f}  "
          f"mean={fb.mean():.4f}")
    print(f"first frame [0:8]: {fb[0, :8].tolist()}")
    print(f"last frame  [0:8]: {fb[-1, :8].tolist()}")

    out = Path(args.out_bin)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        f.write(np.int32(SR).tobytes())
        f.write(np.int32(audio.shape[0]).tobytes())
        f.write(np.int32(N_MELS).tobytes())
        f.write(np.int32(fb.shape[0]).tobytes())
        f.write(audio.tobytes())
        f.write(np.ascontiguousarray(fb.astype(np.float32)).tobytes())

    Path(args.out_meta).write_text(json.dumps({
        "sample_rate":  SR,
        "n_samples":    int(audio.shape[0]),
        "num_mel_bins": N_MELS,
        "n_frames":     int(fb.shape[0]),
        "fb_min":       float(fb.min()),
        "fb_max":       float(fb.max()),
        "fb_mean":      float(fb.mean()),
    }, indent=2), encoding="utf-8")
    print(f"\nWrote {out}  ({out.stat().st_size} bytes)")
    print(f"Wrote {args.out_meta}")


if __name__ == "__main__":
    main()
