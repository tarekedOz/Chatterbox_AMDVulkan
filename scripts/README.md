# scripts/

Python utilities. Run inside the project venv (`.venv/`).

| Script | Purpose | Phase |
|---|---|---|
| `tokenizer_parity_dump.py` | Pulls upstream Turbo tokenizer from HF and emits `tests/tokenizer_groundtruth.json` + `tests/tokenizer_metadata.json`. These pin the oracle our C++ tokenizer must reproduce. | 1 |

## Setup

```pwsh
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/tokenizer_parity_dump.py
```

The HuggingFace cache lives under `%USERPROFILE%\.cache\huggingface` on
Windows. The `tokenizer.json` itself is tiny (~25 kB) — no model weights
are fetched by this script.
