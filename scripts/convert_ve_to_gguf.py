"""
Convert Chatterbox Turbo VoiceEncoder weights into a GGUF file.

The VE is a 3-layer LSTM speaker encoder (1.4M params, fp32 source).
It runs at *conditioning* time only (reference WAV -> 256-d embedding),
never per-token, so we don't need it on the hot path. fp16 is plenty.

Architecture (from upstream models/voice_encoder/voice_encoder.py +
config.py):

    mel(B, T, 40)
       |
       v
    3 x LSTMCell(hidden=256)        # weight_ih_l{0,1,2}, weight_hh_l{0,1,2}
       |
       v
    final_hidden(B, 256)
       |
       v
    Linear(256 -> 256)               # proj.weight, proj.bias
       |
       v
    optional ReLU (ve_final_relu)    # config flag, on by default in Turbo
       |
       v
    L2-normalize over last dim
       |
       v
    speaker_emb(B, 256)

Skipped: similarity_weight, similarity_bias. These are GE2E loss scalars
(init 10.0, -5.0) used only at training time. Inference does not touch
them; including them would only bloat the file by 8 bytes but more
importantly muddy the loader's expectations.

Conversion choices (mirror the T3 converter):

  - Custom arch tag: "chatterbox_ve". Distinct from chatterbox_t3 so the
    same GGUFReader can disambiguate which component is loaded.

  - No tensor reshape/transpose. PyTorch's nn.LSTM stores
    weight_ih_l{N} as [4*hidden, in_size] (gate-stacked i/f/g/o), and
    weight_hh_l{N} as [4*hidden, hidden]. We preserve that layout
    verbatim; chatterbox.cpp's LSTM code will index into the gate
    groups by stride.

  - Default fp16 output, with --dtype f32 escape hatch.

  - Output path defaults to models/chatterbox-turbo-ve-fp16.gguf,
    matching the T3 converter's naming convention.

Usage:
    python scripts/convert_ve_to_gguf.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Force UTF-8 stdout on Windows so non-ASCII status text doesn't cp1252-crash.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from huggingface_hub import hf_hub_download
from safetensors.numpy import load_file

import gguf

REPO_ID = "ResembleAI/chatterbox-turbo"
VE_FILE = "ve.safetensors"
ARCH = "chatterbox_ve"

# --- Architecture constants (from upstream VoiceEncConfig) ---
N_MELS = 40                 # LSTM layer 0 input dim
HIDDEN = 256                # LSTM hidden size, also projection in/out
SPEAKER_EMB_DIM = 256
N_LSTM_LAYERS = 3
SAMPLE_RATE = 16000         # VE-internal sample rate (S3_SR)
FINAL_RELU = True           # ve_final_relu = True in Turbo's config

# State-dict name -> GGUF tensor name.
NAME_MAP_STATIC: dict[str, str] = {
    "proj.weight": "proj.weight",
    "proj.bias":   "proj.bias",
}

NAME_MAP_LSTM: dict[str, str] = {
    "lstm.weight_ih_l{i}": "lstm.l{i}.weight_ih",
    "lstm.weight_hh_l{i}": "lstm.l{i}.weight_hh",
    "lstm.bias_ih_l{i}":   "lstm.l{i}.bias_ih",
    "lstm.bias_hh_l{i}":   "lstm.l{i}.bias_hh",
}

# Training-time GE2E loss scalars — deliberately not written.
SKIP: set[str] = {"similarity_weight", "similarity_bias"}


def build_name_map() -> dict[str, str]:
    m = dict(NAME_MAP_STATIC)
    for i in range(N_LSTM_LAYERS):
        for src_pat, dst_pat in NAME_MAP_LSTM.items():
            m[src_pat.format(i=i)] = dst_pat.format(i=i)
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="models/chatterbox-turbo-ve-fp16.gguf")
    ap.add_argument("--dtype", choices=("f16", "f32"), default="f16")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Fetching {VE_FILE} from {REPO_ID} ...")
    src = Path(hf_hub_download(REPO_ID, VE_FILE))
    print(f"Loading {src.name}")
    state = load_file(str(src))
    print(f"  {len(state)} tensors loaded")

    name_map = build_name_map()

    unmapped = sorted(k for k in state if k not in name_map and k not in SKIP)
    if unmapped:
        print("ERROR: upstream tensors with no mapping:")
        for k in unmapped:
            print(f"  {k}  shape={state[k].shape} dtype={state[k].dtype}")
        sys.exit(1)

    missing = sorted(k for k in name_map if k not in state)
    if missing:
        print("ERROR: expected upstream tensors missing from checkpoint:")
        for k in missing:
            print(f"  {k}")
        sys.exit(1)

    target_dtype = np.float16 if args.dtype == "f16" else np.float32
    file_type = (
        gguf.LlamaFileType.MOSTLY_F16
        if args.dtype == "f16"
        else gguf.LlamaFileType.ALL_F32
    )

    writer = gguf.GGUFWriter(str(out_path), arch=ARCH)

    # --- General metadata ---
    writer.add_name("chatterbox-ve-turbo")
    writer.add_description(
        "Chatterbox Turbo VoiceEncoder (LSTM speaker encoder). "
        "Converted from " + REPO_ID + "."
    )
    writer.add_source_repo_url("https://github.com/resemble-ai/chatterbox")
    writer.add_file_type(file_type)

    # --- Architecture metadata ---
    writer.add_uint32(f"{ARCH}.n_mels", N_MELS)
    writer.add_uint32(f"{ARCH}.hidden_size", HIDDEN)
    writer.add_uint32(f"{ARCH}.speaker_emb_dim", SPEAKER_EMB_DIM)
    writer.add_uint32(f"{ARCH}.lstm_layers", N_LSTM_LAYERS)
    writer.add_uint32(f"{ARCH}.sample_rate", SAMPLE_RATE)
    writer.add_bool(f"{ARCH}.final_relu", FINAL_RELU)
    writer.add_string(f"{ARCH}.tensor_layout", "upstream_native")

    # --- Tensors ---
    # Same convention as the T3 converter: 2D matmul weights cast to target
    # dtype, 1D vectors (biases) stay fp32. Avoids mixed-dtype ggml_add /
    # ggml_mul in the LSTM step (gates + bias) which the CPU backend
    # doesn't support.
    n_written = 0
    n_bytes = 0
    for src_name, dst_name in sorted(name_map.items()):
        t = state[src_name]
        if t.ndim == 1:
            t = t.astype(np.float32)
        else:
            t = t.astype(target_dtype)
        writer.add_tensor(dst_name, t)
        n_written += 1
        n_bytes += t.nbytes

    print(f"  {n_written} tensors mapped -> {n_bytes / (1024 ** 2):.2f} MiB at {args.dtype}")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(f"\nWrote {out_path} ({out_path.stat().st_size / (1024 ** 2):.2f} MiB)")


if __name__ == "__main__":
    main()
