"""
NumPy reference for the C++ host-side resampler.

Uses scipy.signal.resample_poly with the same Kaiser-windowed sinc
filter (beta=8.6, default scipy choice) we'll mirror in C++. The
chosen anti-aliasing cutoff is min(in_sr, out_sr) / 2 / max(in, out)
(i.e., the more-conservative of the two Nyquist rates, normalized).

Test fixtures:
  - 1 second sine sweep + dither, generated at 22050 Hz
  - Resampled to 16000 Hz and 24000 Hz (the two rates the conditioning
    path needs)

Output binary: tests/resampler_reference.bin

    int32   in_sr
    int32   in_len
    float32[in_len]     input_audio
    int32   n_targets
    n_targets * {
        int32   out_sr
        int32   out_len
        float32[out_len]   expected_output
    }
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
from scipy.signal import resample_poly


def make_test_signal(in_sr: int, seconds: float = 1.0,
                       seed: int = 19) -> np.ndarray:
    """Linear sweep from 100 Hz to in_sr/2.5 + small dither."""
    n = int(in_sr * seconds)
    t = np.arange(n, dtype=np.float32) / in_sr
    f0, f1 = 100.0, in_sr / 2.5
    # Linear chirp: f(t) = f0 + (f1 - f0) * t / seconds
    # phase(t) = integral of 2 pi f(t) dt = 2 pi (f0 * t + (f1 - f0) / (2*seconds) * t^2)
    phase = 2.0 * np.pi * (f0 * t + (f1 - f0) / (2.0 * seconds) * t ** 2)
    audio = 0.5 * np.sin(phase).astype(np.float32)
    rng = np.random.RandomState(seed)
    audio += 0.02 * rng.standard_normal(n).astype(np.float32)
    return audio.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-bin",  default="tests/resampler_reference.bin")
    ap.add_argument("--out-meta", default="tests/resampler_reference.json")
    args = ap.parse_args()

    IN_SR = 22050
    audio = make_test_signal(IN_SR, seconds=1.0)
    print(f"Input: {audio.shape[0]} samples @ {IN_SR} Hz "
          f"(min={audio.min():.4f}, max={audio.max():.4f})")

    targets = [16000, 24000]
    outputs = []
    for out_sr in targets:
        from math import gcd
        g = gcd(IN_SR, out_sr)
        up = out_sr // g
        down = IN_SR // g
        # scipy default Kaiser beta is 5.0 but with cutoff and length
        # adjusted; for parity we pass `window=('kaiser', 8.6)` to match
        # the librosa / soxr "high quality" defaults. The filter length
        # is taken from scipy's default (10 * max(up, down) + 1).
        y = resample_poly(audio, up, down,
                            window=("kaiser", 8.6)).astype(np.float32)
        print(f"  -> {out_sr} Hz: {y.shape[0]} samples  "
              f"(up={up}, down={down})  min={y.min():.4f} max={y.max():.4f}")
        outputs.append((out_sr, y))

    out_bin = Path(args.out_bin)
    out_bin.parent.mkdir(parents=True, exist_ok=True)
    with open(out_bin, "wb") as f:
        f.write(np.int32(IN_SR).tobytes())
        f.write(np.int32(audio.shape[0]).tobytes())
        f.write(audio.tobytes())
        f.write(np.int32(len(outputs)).tobytes())
        for out_sr, y in outputs:
            f.write(np.int32(out_sr).tobytes())
            f.write(np.int32(y.shape[0]).tobytes())
            f.write(y.tobytes())

    Path(args.out_meta).write_text(json.dumps({
        "in_sr": IN_SR,
        "in_len": int(audio.shape[0]),
        "targets": [{"out_sr": sr, "out_len": int(y.shape[0])} for sr, y in outputs],
    }, indent=2), encoding="utf-8")
    print(f"\nWrote {out_bin}  ({out_bin.stat().st_size} bytes)")
    print(f"Wrote {args.out_meta}")


if __name__ == "__main__":
    main()
