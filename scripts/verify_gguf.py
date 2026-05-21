"""
Verify that a converted Chatterbox GGUF round-trips against the upstream
safetensors. Auto-detects which component the GGUF is for by reading the
arch metadata key from the header, then dispatches to the matching
upstream file and tensor mapping.

Supported archs:
    chatterbox_t3     -> t3_turbo_v1.safetensors
    chatterbox_ve     -> ve.safetensors
    chatterbox_s3gen  -> s3gen_meanflow.safetensors

Tolerance: atol=1e-3, rtol=1e-3 — fp16 round-trip loses precision in
the LSB; bit-equality is not achievable.

Exits non-zero on any mismatch.

Usage:
    python scripts/verify_gguf.py models/chatterbox-turbo-t3-fp16.gguf
    python scripts/verify_gguf.py models/chatterbox-turbo-ve-fp16.gguf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# UTF-8 stdout on Windows (cp1252 trips on non-ASCII).
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from huggingface_hub import hf_hub_download
from safetensors.numpy import load_file

import gguf

REPO_ID = "ResembleAI/chatterbox-turbo"

ATOL = 1e-3
RTOL = 1e-3
N_LAYERS_T3 = 24
N_LSTM_LAYERS_VE = 3

# S3Gen uses identity tensor naming with one filter: BN counters skipped.
def _s3gen_inverse_map(upstream_state: dict[str, np.ndarray]) -> dict[str, str]:
    return {
        name: name
        for name in upstream_state
        if not name.endswith(".num_batches_tracked")
    }


def _t3_inverse_map() -> dict[str, str]:
    static = {
        "token_embd.text.weight":   "text_emb.weight",
        "token_embd.speech.weight": "speech_emb.weight",
        "text_head.weight":         "text_head.weight",
        "speech_head.weight":       "speech_head.weight",
        "speech_head.bias":         "speech_head.bias",
        "cond_enc.weight":          "cond_enc.spkr_enc.weight",
        "cond_enc.bias":            "cond_enc.spkr_enc.bias",
        "position_embd.weight":     "tfmr.wpe.weight",
        "output_norm.weight":       "tfmr.ln_f.weight",
        "output_norm.bias":         "tfmr.ln_f.bias",
    }
    layer = {
        "blk.{i}.attn_norm.weight":   "tfmr.h.{i}.ln_1.weight",
        "blk.{i}.attn_norm.bias":     "tfmr.h.{i}.ln_1.bias",
        "blk.{i}.attn_qkv.weight":    "tfmr.h.{i}.attn.c_attn.weight",
        "blk.{i}.attn_qkv.bias":      "tfmr.h.{i}.attn.c_attn.bias",
        "blk.{i}.attn_output.weight": "tfmr.h.{i}.attn.c_proj.weight",
        "blk.{i}.attn_output.bias":   "tfmr.h.{i}.attn.c_proj.bias",
        "blk.{i}.ffn_norm.weight":    "tfmr.h.{i}.ln_2.weight",
        "blk.{i}.ffn_norm.bias":      "tfmr.h.{i}.ln_2.bias",
        "blk.{i}.ffn_up.weight":      "tfmr.h.{i}.mlp.c_fc.weight",
        "blk.{i}.ffn_up.bias":        "tfmr.h.{i}.mlp.c_fc.bias",
        "blk.{i}.ffn_down.weight":    "tfmr.h.{i}.mlp.c_proj.weight",
        "blk.{i}.ffn_down.bias":      "tfmr.h.{i}.mlp.c_proj.bias",
    }
    m = dict(static)
    for i in range(N_LAYERS_T3):
        for k, v in layer.items():
            m[k.format(i=i)] = v.format(i=i)
    return m


def _ve_inverse_map() -> dict[str, str]:
    static = {
        "proj.weight": "proj.weight",
        "proj.bias":   "proj.bias",
    }
    layer = {
        "lstm.l{i}.weight_ih": "lstm.weight_ih_l{i}",
        "lstm.l{i}.weight_hh": "lstm.weight_hh_l{i}",
        "lstm.l{i}.bias_ih":   "lstm.bias_ih_l{i}",
        "lstm.l{i}.bias_hh":   "lstm.bias_hh_l{i}",
    }
    m = dict(static)
    for i in range(N_LSTM_LAYERS_VE):
        for k, v in layer.items():
            m[k.format(i=i)] = v.format(i=i)
    return m


# Arch tag -> (upstream filename, inverse-map builder).
# Builders that take no args produce a static map; the special sentinel
# 'data_dependent' means "call with the upstream state dict" (used by
# s3gen, where naming is identity-with-filter and the filter only makes
# sense relative to actual upstream keys).
COMPONENTS: dict[str, tuple[str, object]] = {
    "chatterbox_t3":    ("t3_turbo_v1.safetensors", _t3_inverse_map),
    "chatterbox_ve":    ("ve.safetensors",          _ve_inverse_map),
    "chatterbox_s3gen": ("s3gen_meanflow.safetensors", _s3gen_inverse_map),
}


def read_arch(reader: gguf.GGUFReader) -> str:
    field = reader.get_field("general.architecture")
    if field is None:
        sys.exit("GGUF is missing general.architecture key.")
    # Field.parts is a list of numpy arrays; string is stored in the last part.
    return str(bytes(field.parts[-1]).decode("utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf_path", type=Path)
    ap.add_argument("--sample-size", type=int, default=512,
                    help="Random elements per tensor to sample for value check")
    args = ap.parse_args()

    if not args.gguf_path.exists():
        sys.exit(f"GGUF not found: {args.gguf_path}")

    print(f"Reading GGUF: {args.gguf_path}")
    reader = gguf.GGUFReader(str(args.gguf_path))
    arch = read_arch(reader)
    print(f"Detected arch: {arch}")

    if arch not in COMPONENTS:
        sys.exit(f"Unknown chatterbox arch: {arch}. Known: {sorted(COMPONENTS)}")

    upstream_file, build_map = COMPONENTS[arch]

    print(f"Loading upstream {upstream_file} ...")
    src = Path(hf_hub_download(REPO_ID, upstream_file))
    upstream = load_file(str(src))

    # Data-dependent builders (s3gen) accept the upstream state dict;
    # static builders (t3, ve) take nothing. Dispatch by signature.
    import inspect
    sig = inspect.signature(build_map)
    inv_map = build_map(upstream) if sig.parameters else build_map()

    gguf_tensors = {t.name: t for t in reader.tensors}

    print(f"\nGGUF tensors: {len(gguf_tensors)}")
    print(f"Mapping entries: {len(inv_map)}")

    fails: list[str] = []
    rng = np.random.default_rng(seed=0)

    missing_in_gguf = sorted(set(inv_map) - set(gguf_tensors))
    extra_in_gguf = sorted(set(gguf_tensors) - set(inv_map))
    if missing_in_gguf:
        fails.append(f"Missing from GGUF: {missing_in_gguf}")
    if extra_in_gguf:
        fails.append(f"Unexpected in GGUF: {extra_in_gguf}")

    n_checked = 0
    max_abs_err = 0.0
    for gguf_name, upstream_name in sorted(inv_map.items()):
        if gguf_name not in gguf_tensors:
            continue
        gt = gguf_tensors[gguf_name]
        up = upstream[upstream_name]

        gguf_shape = tuple(int(d) for d in reversed(gt.shape))
        if gguf_shape != tuple(up.shape):
            if tuple(int(d) for d in gt.shape) != tuple(up.shape):
                fails.append(
                    f"Shape mismatch {gguf_name}: gguf={tuple(gt.shape)} upstream={up.shape}"
                )
                continue

        gguf_data = np.asarray(gt.data).reshape(up.shape).astype(np.float32)
        up_f32 = up.astype(np.float32)

        n = up_f32.size
        k = min(args.sample_size, n)
        idx = rng.choice(n, size=k, replace=False) if n > k else np.arange(n)
        diff = np.abs(gguf_data.flat[idx] - up_f32.flat[idx])
        max_err = float(diff.max(initial=0.0))
        max_abs_err = max(max_abs_err, max_err)

        threshold = ATOL + RTOL * np.abs(up_f32.flat[idx]).max(initial=0.0)
        if max_err > threshold:
            fails.append(
                f"Value drift {gguf_name}: max_abs_err={max_err:.2e} threshold={threshold:.2e}"
            )

        n_checked += 1

    print(f"Checked {n_checked} tensors, max abs error across samples: {max_abs_err:.2e}")

    if fails:
        print("\nFAILURES:")
        for f in fails:
            print(f"  {f}")
        sys.exit(1)
    print("\nOK")


if __name__ == "__main__":
    main()
