"""Whisper - Speech Recognition (Reference Sec 47.231).

Radford, Kim, Xu, Brockman, McLeavey & Sutskever 2022 'Robust
Speech Recognition via Large-Scale Weak Supervision', OpenAI /
Interspeech 2023. Encoder-decoder Transformer trained on 680,000
hours of weakly-labelled internet audio:

    Audio -> log-Mel spectrogram (80 x T)
    -> Transformer ENCODER (self-attn over time)
    -> Transformer DECODER (cross-attn to encoder + causal self-attn over tokens)
    -> transcription in the LANGUAGE'S SCRIPT with punctuation.

Handles 99 languages, translates to English, timestamps and
paralinguistic tokens ([BGM], [applause]).
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


LANG_TOKENS = ["<en>", "<fr>", "<de>", "<es>"]
TASK_TOKENS = ["<transcribe>", "<translate>"]


def mel_spectrogram(waveform, n_mels=80, n_frames=100):
    L = len(waveform)
    frame_len = max(1, L // n_frames)
    spec = np.zeros((n_mels, n_frames))
    for t in range(n_frames):
        start = t * frame_len
        segment = waveform[start:start + frame_len * 2]
        if len(segment) < 2: continue
        F = np.abs(np.fft.rfft(segment))[:n_mels]
        spec[:len(F), t] = np.log(F + 1e-6)
    return spec


def toy_encoder(mel, dim=64, rng=None):
    """Toy encoder: linear pool over time."""
    if rng is None: rng = np.random.default_rng(0)
    n_mels, T = mel.shape
    W = rng.normal(scale=0.1, size=(n_mels, dim))
    return (mel.T @ W)                                            # (T, dim)


def toy_decode(encoder_out, prompt_tokens, vocab, W_dec, max_len=20):
    """Toy autoregressive decoder using nearest-neighbor over vocab embeddings."""
    tokens = list(prompt_tokens)
    for step in range(max_len):
        # Context vector: mean of encoder + embedding of previous tokens
        ctx = encoder_out.mean(axis=0)
        for t in tokens[-3:]:
            ctx = ctx + W_dec[vocab.get(t, 0)]
        # Score each vocab entry by similarity to context
        scores = np.array([W_dec[i] @ ctx for i in range(len(vocab))])
        idx_to_tok = {i: t for t, i in vocab.items()}
        next_tok = idx_to_tok[int(np.argmax(scores))]
        if next_tok == "<eos>": break
        tokens.append(next_tok)
    return tokens


if __name__ == "__main__":
    print("=== Whisper (Radford et al 2022) ===\n")
    rng = np.random.default_rng(0)

    sr = 8000; T = 1.0
    t = np.arange(0, T, 1 / sr)
    # Toy 'speech': tone burst + noise
    waveform = 0.5 * np.sin(2 * np.pi * 300 * t) * (t < 0.5) + 0.1 * rng.uniform(-1, 1, size=len(t))
    mel = mel_spectrogram(waveform, n_mels=80, n_frames=100)
    print(f"  Waveform: {len(waveform)} samples at {sr} Hz -> mel {mel.shape}")

    # Toy vocab
    vocab_list = ["<en>", "<fr>", "<transcribe>", "<translate>", "hello", "world", "bonjour",
                    "monde", "the", "quick", "<eos>"]
    vocab = {t: i for i, t in enumerate(vocab_list)}
    W_dec = rng.normal(scale=0.3, size=(len(vocab), 64))

    enc_out = toy_encoder(mel, dim=64, rng=rng)

    # Task 1: transcribe English
    tokens = toy_decode(enc_out, ["<en>", "<transcribe>"], vocab, W_dec, max_len=8)
    print(f"  Prompt = [<en>, <transcribe>] -> {tokens}")

    # Task 2: translate (French)
    tokens = toy_decode(enc_out, ["<fr>", "<translate>"], vocab, W_dec, max_len=8)
    print(f"  Prompt = [<fr>, <translate>] -> {tokens}")

    print("\n  Real Whisper (large-v3) achieves ~5% WER on LibriSpeech and handles")
    print("  99 languages, translation to English, and timestamps.")

    print("\n--- library cross-check (openai-whisper; faster-whisper; distil-whisper) ---")
