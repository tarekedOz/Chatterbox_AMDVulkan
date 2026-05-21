"""
Checkpoint inventory for Chatterbox Turbo.

Downloads each safetensors weight file from ResembleAI/chatterbox-turbo
(skipping the non-meanflow s3gen variant — we use the meanflow one), opens
each via safetensors' metadata-only path (no torch needed), and emits a
JSON inventory of every tensor: name, shape, dtype, byte size, and
inferred parameter count.

The inventory is the ground truth for designing convert_chatterbox_to_gguf.py.
Without it, the converter would be speculation about state-dict layout.

Output:
  - docs/checkpoint-inventory.json
"""

from __future__ import annotations

import json
from pathlib import Path

from huggingface_hub import hf_hub_download
from safetensors import safe_open

REPO_ID = "ResembleAI/chatterbox-turbo"
ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "docs" / "checkpoint-inventory.json"

# Per tts_turbo.py: from_local() loads ve.safetensors, t3_turbo_v1.safetensors,
# and s3gen_meanflow.safetensors. The non-meanflow s3gen.safetensors is an
# alternate not used by the meanflow Turbo path — skip it (saves 1 GB download
# we'd never use).
WEIGHT_FILES = [
    "ve.safetensors",
    "t3_turbo_v1.safetensors",
    "s3gen_meanflow.safetensors",
]

# safetensors dtype string -> bytes per element.
DTYPE_BYTES = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2,
    "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1,
    "BOOL": 1,
}


def inspect(path: Path) -> dict:
    tensors = []
    total_bytes = 0
    total_params = 0
    with safe_open(str(path), framework="numpy") as f:
        for name in f.keys():
            t = f.get_slice(name)
            shape = list(t.get_shape())
            dtype = t.get_dtype()
            numel = 1
            for d in shape:
                numel *= d
            bytes_per = DTYPE_BYTES.get(dtype, 0)
            nbytes = numel * bytes_per
            tensors.append({
                "name": name,
                "shape": shape,
                "dtype": dtype,
                "numel": numel,
                "bytes": nbytes,
            })
            total_bytes += nbytes
            total_params += numel
    tensors.sort(key=lambda x: x["name"])
    return {
        "file": path.name,
        "file_bytes": path.stat().st_size,
        "tensor_count": len(tensors),
        "total_params": total_params,
        "total_weight_bytes": total_bytes,
        "tensors": tensors,
    }


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    inventory = {"source_repo": REPO_ID, "files": []}

    for fname in WEIGHT_FILES:
        print(f"Fetching {fname} ...")
        local = Path(hf_hub_download(REPO_ID, fname))
        print(f"  -> {local}")
        info = inspect(local)
        inventory["files"].append(info)
        print(
            f"  {info['tensor_count']} tensors, "
            f"{info['total_params'] / 1e6:.1f}M params, "
            f"{info['total_weight_bytes'] / (1024 ** 2):.1f} MiB"
        )

    OUT_PATH.write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    print(f"\nWrote {OUT_PATH}")

    grand_params = sum(f["total_params"] for f in inventory["files"])
    grand_bytes = sum(f["total_weight_bytes"] for f in inventory["files"])
    print(f"Total: {grand_params / 1e6:.1f}M params across {len(inventory['files'])} files "
          f"({grand_bytes / (1024 ** 3):.2f} GiB)")


if __name__ == "__main__":
    main()
