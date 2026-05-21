"""
Convert Chatterbox Turbo S3Gen weights into a GGUF file.

S3Gen is the speech-token -> waveform path. It comprises four submodules:

    tokenizer          S3Tokenizer ("speech_tokenizer_v2_25hz")
                       Encodes a reference WAV into 6563-vocab discrete
                       speech tokens at conditioning time. 124M params.

    speaker_encoder    CAMPPlus (x-vector style)
                       Produces a speaker embedding from the reference
                       WAV's mel spectrogram. Distinct from VE (used by
                       T3); only consumed by the flow encoder. 7M params.

    flow               CausalMaskedDiffWithXvec
                         encoder  = UpsampleConformerEncoder (8h, 6 blk)
                         decoder  = CausalConditionalCFM wrapping
                                    ConditionalDecoder (UNet,
                                    320 -> 80 channels, 4 + 12 blk).
                       Speech-token -> mel via conditional flow matching.
                       115M params total across encoder + decoder.

    mel2wav            HiFTGenerator (HiFiGAN + ConvRNN F0 predictor
                       + harmonic source modulation).
                       Mel -> 24 kHz waveform. 21M params.

This is the "meanflow" variant of S3Gen, which reduces the ODE solver
to 2 timesteps (vs 10 in the non-meanflow s3gen.safetensors).
chatterbox.cpp will need this 2-step short-cut wired in to hit the
Phase-2 <2s latency target.

Conversion choices:

  - Custom arch tag: "chatterbox_s3gen".

  - Identity tensor renaming. The upstream prefixes (tokenizer.*,
    flow.encoder.*, flow.decoder.*, speaker_encoder.*, mel2wav.*) are
    already structured and unique within the file. We preserve names
    verbatim. The C++ loader will own all semantic interpretation.

  - Skip every tensor with suffix `.num_batches_tracked`. PyTorch
    BatchNorm tracks this for training; at inference it's dead weight.
    All 122 I64 tensors in the upstream safetensors fall under this.

  - Preserve weight_norm parametrizations as-is. Conv layers using
    nn.utils.weight_norm store the weight as a (magnitude, direction)
    pair under `.parametrizations.weight.original0` (g, shape [C,1,1])
    and `.original1` (v, shape [C,K,...]). chatterbox.cpp can either
    pre-combine at load (weight = g * v / ||v||) or apply on the fly.

  - Default fp16. Output: models/chatterbox-turbo-s3gen-fp16.gguf
    (~520 MiB).

Usage:
    python scripts/convert_s3gen_to_gguf.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# UTF-8 stdout on Windows.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from huggingface_hub import hf_hub_download
from safetensors.numpy import load_file

import gguf

REPO_ID = "ResembleAI/chatterbox-turbo"
S3GEN_FILE = "s3gen_meanflow.safetensors"
ARCH = "chatterbox_s3gen"

# --- Architecture constants (mirrored from upstream models/s3gen/) ---
# These let the C++ loader sanity-check what it's reading.
S3GEN_SR = 24000              # output waveform sample rate
S3_SR = 16000                 # tokenizer-side input sample rate
S3GEN_SIL = 4299              # silence token id (appended after T3 output)
SPEECH_VOCAB = 6563           # speech token vocabulary

FLOW_ENCODER_DIM = 512
FLOW_ENCODER_HEADS = 8
FLOW_ENCODER_BLOCKS = 6
FLOW_DECODER_IN_CHANNELS = 320
FLOW_DECODER_OUT_CHANNELS = 80
FLOW_DECODER_BLOCKS = 4
FLOW_DECODER_MID_BLOCKS = 12
N_CFM_TIMESTEPS = 2           # meanflow shortcut (vs. 10 in vanilla S3Gen)
TOKENIZER_NAME = "speech_tokenizer_v2_25hz"


def is_skip(name: str) -> bool:
    # BatchNorm training-state counters. 122 of these in the upstream
    # checkpoint, all I64 dtype, no inference value.
    return name.endswith(".num_batches_tracked")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="models/chatterbox-turbo-s3gen-fp16.gguf")
    ap.add_argument("--dtype", choices=("f16", "f32"), default="f16")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Fetching {S3GEN_FILE} from {REPO_ID} ...")
    src = Path(hf_hub_download(REPO_ID, S3GEN_FILE))
    print(f"Loading {src.name}")
    state = load_file(str(src))
    print(f"  {len(state)} tensors loaded")

    # Identity mapping with skip filter.
    target_dtype = np.float16 if args.dtype == "f16" else np.float32
    file_type = (
        gguf.LlamaFileType.MOSTLY_F16
        if args.dtype == "f16"
        else gguf.LlamaFileType.ALL_F32
    )

    writer = gguf.GGUFWriter(str(out_path), arch=ARCH)

    # --- General metadata ---
    writer.add_name("chatterbox-s3gen-turbo")
    writer.add_description(
        "Chatterbox Turbo S3Gen (speech-token -> waveform). "
        "Meanflow variant. Converted from " + REPO_ID + "."
    )
    writer.add_source_repo_url("https://github.com/resemble-ai/chatterbox")
    writer.add_file_type(file_type)

    # --- Architecture metadata ---
    writer.add_uint32(f"{ARCH}.output_sample_rate", S3GEN_SR)
    writer.add_uint32(f"{ARCH}.tokenizer_sample_rate", S3_SR)
    writer.add_uint32(f"{ARCH}.silence_token_id", S3GEN_SIL)
    writer.add_uint32(f"{ARCH}.speech_vocab_size", SPEECH_VOCAB)
    writer.add_uint32(f"{ARCH}.flow_encoder_dim", FLOW_ENCODER_DIM)
    writer.add_uint32(f"{ARCH}.flow_encoder_heads", FLOW_ENCODER_HEADS)
    writer.add_uint32(f"{ARCH}.flow_encoder_blocks", FLOW_ENCODER_BLOCKS)
    writer.add_uint32(f"{ARCH}.flow_decoder_in_channels", FLOW_DECODER_IN_CHANNELS)
    writer.add_uint32(f"{ARCH}.flow_decoder_out_channels", FLOW_DECODER_OUT_CHANNELS)
    writer.add_uint32(f"{ARCH}.flow_decoder_blocks", FLOW_DECODER_BLOCKS)
    writer.add_uint32(f"{ARCH}.flow_decoder_mid_blocks", FLOW_DECODER_MID_BLOCKS)
    writer.add_uint32(f"{ARCH}.cfm_timesteps", N_CFM_TIMESTEPS)
    writer.add_bool(f"{ARCH}.meanflow", True)
    writer.add_string(f"{ARCH}.tokenizer_name", TOKENIZER_NAME)
    writer.add_string(f"{ARCH}.tensor_layout", "upstream_native")

    # --- Tensors ---
    n_written = 0
    n_skipped = 0
    n_bytes = 0
    for name in sorted(state.keys()):
        if is_skip(name):
            n_skipped += 1
            continue
        t = state[name]
        if t.dtype not in (np.float32, np.float16, np.float64):
            # Defensive — by inventory we should have already filtered all
            # I64s via the suffix rule. Anything else slipping through is
            # a bug worth surfacing rather than silently corrupting.
            print(f"WARNING: unexpected dtype for {name}: {t.dtype}; skipping")
            n_skipped += 1
            continue
        # 1D vectors stay fp32 (biases, layernorm gains, etc.); 2D+ weights
        # get the user-chosen target dtype. Same convention as T3 / VE.
        #
        # Special-case: the tokenizer's mel filterbank + Hann window are
        # consumed by host-side mel-extraction code (chatterbox-cpp/src/mel.cpp),
        # not by any ggml matmul. Keep them fp32 always — fp16 here costs
        # ~1e-4 of mel filter precision which the log normalization then
        # amplifies into ~1e-2 drift at the noise floor of log-mel.
        if name in ("tokenizer._mel_filters", "tokenizer.window"):
            t = t.astype(np.float32)
        elif t.ndim == 1:
            t = t.astype(np.float32)
        else:
            t = t.astype(target_dtype)
        writer.add_tensor(name, t)
        n_written += 1
        n_bytes += t.nbytes

    print(f"  wrote {n_written} tensors, skipped {n_skipped}; "
          f"{n_bytes / (1024 ** 2):.1f} MiB at {args.dtype}")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(f"\nWrote {out_path} ({out_path.stat().st_size / (1024 ** 2):.1f} MiB)")


if __name__ == "__main__":
    main()
