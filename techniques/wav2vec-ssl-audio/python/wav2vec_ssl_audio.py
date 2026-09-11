"""wav2vec 2.0 - Self-Supervised Speech Representations (Sec 47.232).

Baevski, Zhou, Mohamed & Auli 2020 'wav2vec 2.0: A Framework for
Self-Supervised Learning of Speech Representations', NeurIPS.
Two-stage:

    1. CNN feature encoder: raw waveform -> latent z_1..z_T.
    2. TRANSFORMER context net: mask spans of z_t and predict via
       CONTRASTIVE loss against a set of QUANTISED targets q_t
       (a small learned codebook via Gumbel-softmax).

L_contrastive = -log( exp(sim(c_t, q_t) / kappa) / sum_qhat exp(sim(c_t, qhat) / kappa) )

Enables strong ASR with 10 min of labelled data + 60k hours of
unlabelled (LibriLight). Direct ancestor of Whisper's audio
encoder philosophy.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def toy_cnn_encoder(waveform, out_dim=64, stride=100):
    """Toy 'CNN' encoder: mean-pool over local windows -> feature vectors."""
    T = len(waveform) // stride
    frames = np.array([waveform[i * stride:(i + 1) * stride] for i in range(T)])
    W = np.random.default_rng(0).normal(scale=0.1, size=(stride, out_dim))
    return frames @ W


def quantise(z, codebook):
    """Codebook lookup: return index of nearest codebook entry."""
    dists = np.linalg.norm(codebook[None, :] - z[:, None], axis=-1)
    idx = np.argmin(dists, axis=-1)
    return idx, codebook[idx]


def mask_spans(z, mask_prob=0.5, span_len=3, rng=None):
    """Randomly mask spans of consecutive frames; return mask boolean."""
    if rng is None: rng = np.random.default_rng(0)
    T = len(z)
    mask = np.zeros(T, dtype=bool)
    n_starts = int(mask_prob * T / span_len)
    starts = rng.choice(T - span_len, size=n_starts, replace=False)
    for s in starts:
        mask[s:s + span_len] = True
    return mask


def contrastive_loss(context, target_q, distractors, tau=0.1):
    """L2-normalised contrastive: pull context toward its own target, push away distractors."""
    context_n = context / (np.linalg.norm(context, axis=-1, keepdims=True) + 1e-8)
    target_n = target_q / (np.linalg.norm(target_q, axis=-1, keepdims=True) + 1e-8)
    distr_n = distractors / (np.linalg.norm(distractors, axis=-1, keepdims=True) + 1e-8)
    pos = np.sum(context_n * target_n, axis=-1) / tau
    neg = context_n @ distr_n.T / tau
    denom = np.exp(pos)[:, None] + np.exp(neg).sum(axis=-1, keepdims=True)
    return float(np.mean(-pos + np.log(denom.squeeze() + 1e-12)))


if __name__ == "__main__":
    print("=== wav2vec 2.0 (Baevski et al 2020 NeurIPS) ===\n")
    rng = np.random.default_rng(0)

    sr = 8000; T_seconds = 1.0
    t = np.arange(0, T_seconds, 1 / sr)
    # Toy 'speech' waveform
    waveform = 0.5 * np.sin(2 * np.pi * 300 * t) + 0.1 * rng.uniform(-1, 1, size=len(t))

    z = toy_cnn_encoder(waveform, out_dim=64, stride=100)
    print(f"  Waveform ({len(waveform)} samples at {sr} Hz) -> {z.shape} latent features")

    # Codebook (learned quantiser); 256 codes here
    codebook = rng.normal(scale=0.5, size=(256, 64))
    idx, q = quantise(z, codebook)
    print(f"  Quantised via 256-entry codebook: {len(np.unique(idx))} distinct codes used")

    # Mask spans and compute contrastive loss
    mask = mask_spans(z, mask_prob=0.5, span_len=3, rng=rng)
    print(f"  Masked {int(mask.sum())} / {len(z)} frames")

    # 'Context network' output (toy: identity)
    context = z[mask]
    target_q = q[mask]
    # Distractors: 100 random codebook entries
    distractors = codebook[rng.choice(256, size=100, replace=False)]
    L = contrastive_loss(context, target_q, distractors, tau=0.1)
    print(f"  Contrastive loss (masked pos vs 100 distractors): {L:.3f}")

    # Random baseline: swap targets
    scrambled_q = q[mask][rng.permutation(mask.sum())]
    L_random = contrastive_loss(context, scrambled_q, distractors, tau=0.1)
    print(f"  Contrastive loss with RANDOM targets:              {L_random:.3f}  (should be higher)")

    print("\n  Real wav2vec 2.0 pretrains on 60k hrs of raw audio, then fine-tunes on")
    print("  ~10 min of labelled speech to hit ASR WER competitive with fully-supervised.")

    print("\n--- library cross-check (facebookresearch/fairseq/wav2vec; transformers.Wav2Vec2Model) ---")
