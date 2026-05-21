"""
Pack the 28 pre-shipped Chatterbox voices into a sidecar file.

STATUS: **BLOCKED** — this is a Phase 1.D follow-up. Run requirements
are documented below; the script exits early today with a pointer.

# What voices.bin needs to contain

Per upstream's `Conditionals` dataclass (see
chatterbox/src/chatterbox/tts_turbo.py), each voice's conditioning is
two coupled blobs:

    T3Cond (consumed by t3.inference_turbo):
      - speaker_emb                256-d, from VoiceEncoder.embeds_from_wavs
      - cond_prompt_speech_tokens  ~375 tokens, from S3Gen.tokenizer
      - emotion_adv                scalar (exaggeration), runtime-set

    Gen ref dict (consumed by s3gen.inference):
      - prompt_token               from S3Gen.tokenizer
      - prompt_token_len           scalar
      - prompt_feat                mel features
      - prompt_feat_len            scalar
      - embedding                  from S3Gen.speaker_encoder (CAMPPlus)

Upstream's `Conditionals.save(...)` serializes this via torch.save into
a ~169 kB pickle (see ResembleAI/chatterbox-turbo/conds.pt for the
default-voice instance). For 28 voices: ~5 MiB total. Small.

# Why this is blocked today

To produce one Conditionals blob from a reference WAV requires running:

    1. VoiceEncoder (LSTM speaker encoder)
    2. S3Tokenizer (`speech_tokenizer_v2_25hz`) — encode ref to tokens
    3. S3Gen.speaker_encoder (CAMPPlus) — separate speaker embedding
    4. The s3gen.embed_ref pipeline (mel extraction + speaker vector)

We have GGUF weights for all of these (chatterbox-turbo-{ve,s3gen}-fp16.gguf)
but **no inference runtime** yet — chatterbox.cpp's forward passes are a
Phase 1.E deliverable.

Two ways to unblock voices.bin:

    A) Install PyTorch + the upstream chatterbox-tts package once,
       run Conditionals.save() per voice, ship the resulting blobs.
       Pro: cheap, works today.
       Cons: pulls ~2 GB of torch + cuda deps into the dev env;
             couples our build to a Python toolchain we deliberately
             avoided.

    B) Wait until chatterbox.cpp can run the conditioning path
       (VE + S3 tokenizer + CAMPPlus + mel ext) in pure C++, then
       extract from there.
       Pro: self-contained; matches the production runtime exactly.
       Cons: gates v1 release on Phase 1.E completing.

Decision deferred to when v1 packaging starts. The 28 reference WAVs
ship with devnen/Chatterbox-TTS-Server under `voices/*.wav`.

# Schema we'll write (when ready)

Plain GGUF, custom arch "chatterbox_voices". Per voice:

    voices.{i}.id                  string, e.g. "Abigail"
    voices.{i}.t3_speaker_emb      F16 tensor [256]
    voices.{i}.t3_prompt_tokens    I32 tensor [375]
    voices.{i}.s3gen_prompt_token  I32 tensor [N]
    voices.{i}.s3gen_prompt_feat   F16 tensor [N, n_mels]
    voices.{i}.s3gen_embedding     F16 tensor [..]

Plus a top-level array `voices.list = [str, ...]` for fast iteration.
"""

import sys


def main() -> int:
    print(__doc__, file=sys.stderr)
    print("\nNot runnable yet. See docs/session-state.md task tracking.",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
