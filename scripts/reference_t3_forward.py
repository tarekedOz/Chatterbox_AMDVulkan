"""
NumPy reference forward pass for Chatterbox T3.

Phase 1.E parity oracle. Loads t3_turbo_v1.safetensors and runs a
stripped-down forward through the GPT-2-medium backbone:

    text_emb[ids] + pos_embd[0..L)
    -> 24 x (LN -> causal MHA -> +residual -> LN -> FFN -> +residual)
    -> output LN
    -> speech_head @ last_position
    -> logits[6563]

No conditioning, no KV cache, no AR loop, no sampling. Just the
forward math, in fp32 NumPy. The C++ implementation must reproduce
these logits within fp16 round-trip tolerance.

Output: tests/t3_reference.bin (binary):
    int32   n_tokens
    int32 * n_tokens   token_ids
    float32 * 6563     logits at position n_tokens - 1
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
T3_FILE = "t3_turbo_v1.safetensors"

N_LAYERS = 24
N_HEADS = 16
EMBED_DIM = 1024
HEAD_DIM = EMBED_DIM // N_HEADS
FFN_DIM = 4096
TEXT_VOCAB = 50276
SPEECH_VOCAB = 6563
SPEAKER_EMB_DIM = 256
LN_EPS = 1e-5

# A fixed test input. These are the token ids for "Hello world." per the
# Phase 1.A tokenizer ground truth (tests/tokenizer_groundtruth.json).
TEST_TOKENS = [15496, 995, 13]

# Deterministic speaker embedding for parity testing. Real Chatterbox
# generation gets this from VoiceEncoder; we don't need a real one for
# the math-correctness test — just any reproducible 256-d vector.
def make_speaker_emb() -> np.ndarray:
    rng = np.random.RandomState(42)
    return rng.standard_normal(SPEAKER_EMB_DIM).astype(np.float32)


# Deterministic prompt speech tokens for parity testing. Real Chatterbox
# gets these from the S3 audio tokenizer over a reference WAV (upstream
# uses speech_cond_prompt_len = 375). We use a small fixed set here —
# the forward math is the same regardless of provenance.
N_PROMPT_TOKENS = 8
def make_cond_prompt_speech_tokens() -> list[int]:
    rng = np.random.RandomState(11)
    return rng.randint(0, 6561, size=N_PROMPT_TOKENS).astype(np.int32).tolist()


def gelu_new(x: np.ndarray) -> np.ndarray:
    """GPT-2's tanh-based GELU approximation. Matches HF NewGELU."""
    return 0.5 * x * (1.0 + np.tanh(
        np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))
    ))


def layer_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float) -> np.ndarray:
    """LayerNorm over the last dimension (gain + bias applied)."""
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * weight + bias


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_max = x.max(axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / e.sum(axis=axis, keepdims=True)


def attention_block(x: np.ndarray, qkv_w: np.ndarray, qkv_b: np.ndarray,
                    out_w: np.ndarray, out_b: np.ndarray, n_heads: int) -> np.ndarray:
    """
    Multi-head causal self-attention.

    qkv_w shape (in=1024, out=3072)  — HF Conv1D order: x @ qkv_w
    out_w shape (in=1024, out=1024)  — HF Conv1D order: y @ out_w
    """
    L, D = x.shape
    H = n_heads
    Hd = D // H

    qkv = x @ qkv_w + qkv_b                          # (L, 3*D)
    q = qkv[:, 0 * D : 1 * D]                        # (L, D)
    k = qkv[:, 1 * D : 2 * D]
    v = qkv[:, 2 * D : 3 * D]

    q = q.reshape(L, H, Hd).transpose(1, 0, 2)        # (H, L, Hd)
    k = k.reshape(L, H, Hd).transpose(1, 0, 2)
    v = v.reshape(L, H, Hd).transpose(1, 0, 2)

    scores = q @ k.transpose(0, 2, 1) / np.sqrt(Hd)   # (H, L, L)

    # Causal mask: position i may attend to j <= i.
    causal = np.triu(np.ones((L, L), dtype=bool), k=1)
    scores = np.where(causal[None, :, :], -np.inf, scores)

    attn = softmax(scores, axis=-1)                   # (H, L, L)
    out = attn @ v                                    # (H, L, Hd)
    out = out.transpose(1, 0, 2).reshape(L, D)        # (L, D)
    return out @ out_w + out_b                        # (L, D)


def ffn_block(x: np.ndarray, up_w: np.ndarray, up_b: np.ndarray,
              dn_w: np.ndarray, dn_b: np.ndarray) -> np.ndarray:
    """GPT-2 FFN: Linear -> GELU -> Linear. Both Conv1D layouts."""
    h = x @ up_w + up_b
    h = gelu_new(h)
    return h @ dn_w + dn_b


def t3_forward(state: dict[str, np.ndarray],
               speaker_emb: np.ndarray,
               cond_prompt_speech_token_ids: list[int],
               text_token_ids: list[int]) -> np.ndarray:
    """Run the forward pass; return logits at the last position (shape (6563,)).

    Sequence layout (Turbo, matches upstream T3CondEnc.forward + T3.forward):

        [cond_spkr (1 token from cond_enc(speaker_emb)),
         cond_prompt_speech (N_prompt tokens via speech_emb),
         text             (N_text tokens via text_emb)]

    Positional embeddings indexed [0..L) over the combined sequence.
    GPT2Model's wpe is added externally (the model receives inputs_embeds,
    bypassing its internal token embedding but applying its positional one).
    """
    L_prompt = len(cond_prompt_speech_token_ids)
    L_text   = len(text_token_ids)
    L = 1 + L_prompt + L_text

    # Speaker conditioning: nn.Linear(256 -> 1024). Weight stored (out, in)
    # so apply as speaker_emb @ W.T + b.
    cond_w = state["cond_enc.spkr_enc.weight"]  # (1024, 256)
    cond_b = state["cond_enc.spkr_enc.bias"]    # (1024,)
    cond_spkr = speaker_emb @ cond_w.T + cond_b  # (1024,)

    pieces = [cond_spkr[None, :]]
    if L_prompt > 0:
        # cond_prompt_speech_emb = speech_emb[cond_prompt_speech_tokens].
        # In Turbo (use_perceiver_resampler=False) the embedding passes
        # through cond_enc unchanged and gets concatenated in.
        prompt_h = state["speech_emb.weight"][cond_prompt_speech_token_ids]
        pieces.append(prompt_h)

    text_h = state["text_emb.weight"][text_token_ids]
    pieces.append(text_h)

    h = np.concatenate(pieces, axis=0)                # (L, 1024)
    h = h + state["tfmr.wpe.weight"][:L]              # (L, 1024)

    for i in range(N_LAYERS):
        ln1 = layer_norm(h,
                         state[f"tfmr.h.{i}.ln_1.weight"],
                         state[f"tfmr.h.{i}.ln_1.bias"], LN_EPS)
        h = h + attention_block(
            ln1,
            state[f"tfmr.h.{i}.attn.c_attn.weight"],
            state[f"tfmr.h.{i}.attn.c_attn.bias"],
            state[f"tfmr.h.{i}.attn.c_proj.weight"],
            state[f"tfmr.h.{i}.attn.c_proj.bias"],
            N_HEADS,
        )
        ln2 = layer_norm(h,
                         state[f"tfmr.h.{i}.ln_2.weight"],
                         state[f"tfmr.h.{i}.ln_2.bias"], LN_EPS)
        h = h + ffn_block(
            ln2,
            state[f"tfmr.h.{i}.mlp.c_fc.weight"],
            state[f"tfmr.h.{i}.mlp.c_fc.bias"],
            state[f"tfmr.h.{i}.mlp.c_proj.weight"],
            state[f"tfmr.h.{i}.mlp.c_proj.bias"],
        )

    h = layer_norm(h, state["tfmr.ln_f.weight"],
                   state["tfmr.ln_f.bias"], LN_EPS)
    last = h[-1]                                      # (1024,)

    # speech_head is nn.Linear with weight stored as (out=6563, in=1024).
    # Apply as last @ weight.T to get (6563,).
    return last @ state["speech_head.weight"].T + state["speech_head.bias"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-bin", default="tests/t3_reference.bin")
    ap.add_argument("--out-meta", default="tests/t3_reference.json")
    args = ap.parse_args()

    print(f"Fetching {T3_FILE} ...")
    src = Path(hf_hub_download(REPO_ID, T3_FILE))
    state = load_file(str(src))
    print(f"  loaded {len(state)} tensors")

    speaker_emb = make_speaker_emb()
    prompt = make_cond_prompt_speech_tokens()
    print(f"Forward on:")
    print(f"  speaker_emb:  shape {speaker_emb.shape}")
    print(f"  cond_prompt:  {prompt}  (length {len(prompt)})")
    print(f"  text_tokens:  {TEST_TOKENS}")
    logits = t3_forward(state, speaker_emb, prompt, TEST_TOKENS).astype(np.float32)

    top5 = np.argsort(-logits)[:5]
    print(f"  logits min/max:   {float(logits.min()):+.4f} / {float(logits.max()):+.4f}")
    print(f"  logits[:5]:        {logits[:5]}")
    print(f"  argmax (top-5):    {top5.tolist()}")
    print(f"  logits at top-5:   {logits[top5]}")

    out_bin = Path(args.out_bin)
    out_bin.parent.mkdir(parents=True, exist_ok=True)
    # Binary format v3:
    #   int32  n_text_tokens
    #   int32 * n_text_tokens             text token ids
    #   int32  speaker_emb_dim
    #   float32 * speaker_emb_dim         speaker_emb
    #   int32  n_prompt_tokens
    #   int32 * n_prompt_tokens           cond_prompt_speech_tokens
    #   float32 * speech_vocab            logits at last position
    with open(out_bin, "wb") as f:
        f.write(np.int32(len(TEST_TOKENS)).tobytes())
        f.write(np.array(TEST_TOKENS, dtype=np.int32).tobytes())
        f.write(np.int32(SPEAKER_EMB_DIM).tobytes())
        f.write(speaker_emb.tobytes())
        f.write(np.int32(len(prompt)).tobytes())
        f.write(np.array(prompt, dtype=np.int32).tobytes())
        f.write(logits.tobytes())

    Path(args.out_meta).write_text(
        json.dumps({
            "text_tokens": TEST_TOKENS,
            "speech_vocab": SPEECH_VOCAB,
            "dtype": "f32",
            "top5_ids": top5.tolist(),
            "top5_logits": logits[top5].tolist(),
            "logits_min": float(logits.min()),
            "logits_max": float(logits.max()),
        }, indent=2),
        encoding="utf-8",
    )

    print(f"\nWrote {out_bin} ({out_bin.stat().st_size} bytes)")
    print(f"Wrote {args.out_meta}")


if __name__ == "__main__":
    main()
