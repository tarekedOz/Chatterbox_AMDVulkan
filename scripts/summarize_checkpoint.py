"""
Generate a Markdown summary of docs/checkpoint-inventory.json.

The raw inventory has 2806 tensors — useful to commit as source of truth,
but unreadable as a design reference. This script collapses numeric path
segments ({N}) so e.g. transformer.h.0.attn.* through transformer.h.23.attn.*
fold into a single pattern, and emits docs/checkpoint-summary.md with the
distinct tensor templates per file.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INV = ROOT / "docs" / "checkpoint-inventory.json"
OUT = ROOT / "docs" / "checkpoint-summary.md"


def collapse(name: str) -> str:
    return ".".join("{N}" if p.isdigit() else p for p in name.split("."))


def fmt_shape(shape: list[int]) -> str:
    return "[" + ", ".join(str(s) for s in shape) + "]"


def main() -> None:
    inv = json.loads(INV.read_text(encoding="utf-8"))
    lines: list[str] = []
    lines.append("# Checkpoint Summary — Chatterbox Turbo")
    lines.append("")
    lines.append(
        "*Auto-generated from `docs/checkpoint-inventory.json` by "
        "`scripts/summarize_checkpoint.py`. Do not hand-edit.*"
    )
    lines.append("")
    lines.append(
        f"Source repo: `{inv['source_repo']}`. "
        f"Files: {len(inv['files'])}. "
        f"Grand totals: "
        f"{sum(f['tensor_count'] for f in inv['files'])} tensors, "
        f"{sum(f['total_params'] for f in inv['files'])/1e6:.1f}M params, "
        f"{sum(f['total_weight_bytes'] for f in inv['files'])/(1024**3):.2f} GiB on disk."
    )

    for f in inv["files"]:
        lines.append("")
        lines.append(
            f"## `{f['file']}` ({f['tensor_count']} tensors, "
            f"{f['total_params']/1e6:.1f}M params, "
            f"{f['total_weight_bytes']/(1024**2):.1f} MiB)"
        )
        lines.append("")
        buckets: dict[str, list[dict]] = defaultdict(list)
        for t in f["tensors"]:
            buckets[collapse(t["name"])].append(t)
        lines.append("| Pattern | Count | Shape | Dtype |")
        lines.append("|---|---:|---|---|")
        for pattern, ts in sorted(buckets.items()):
            shape = fmt_shape(ts[0]["shape"])
            dtype = ts[0]["dtype"]
            lines.append(f"| `{pattern}` | {len(ts)} | `{shape}` | {dtype} |")

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
