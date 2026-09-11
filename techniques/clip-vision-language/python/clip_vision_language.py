"""CLIP - Contrastive Language-Image Pre-training (Sec 47.226).

Radford et al 2021 'Learning Transferable Visual Models From
Natural Language Supervision', ICML. Jointly train an image
encoder E_i and text encoder E_t on 400M (image, caption) pairs
with in-batch symmetric contrastive loss:

    sim(i, t) = <E_i(i) / ||E_i(i)||, E_t(t) / ||E_t(t)||>
    L = -log(exp(sim(i, t) / tau) / sum_j exp(sim(i, j) / tau))
        (symmetric: matched pairs on the diagonal)

Enables ZERO-SHOT classification via prompt engineering:
compare image embedding to embeddings of class-name prompts.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def make_encoder(vocab_or_pixels, dim, rng):
    """Toy encoder: linear projection."""
    return rng.normal(scale=0.3, size=(vocab_or_pixels, dim))


def normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def contrastive_loss(sim, temp=0.07):
    """In-batch symmetric contrastive loss (matched pairs on diagonal)."""
    N = sim.shape[0]
    logits = sim / temp
    labels = np.arange(N)
    loss_i2t = -np.log(np.exp(logits[np.arange(N), labels]) / np.exp(logits).sum(axis=1) + 1e-12)
    loss_t2i = -np.log(np.exp(logits[labels, np.arange(N)]) / np.exp(logits).sum(axis=0) + 1e-12)
    return float(0.5 * (loss_i2t + loss_t2i).mean())


def zero_shot_classify(image_emb, class_prompts, text_encoder):
    """Classify an image by argmax cosine to class-prompt embeddings."""
    scores = [image_emb @ normalize(text_encoder[p]) for p in class_prompts]
    return int(np.argmax(scores)), scores


if __name__ == "__main__":
    print("=== CLIP (Radford et al 2021 ICML) ===\n")
    rng = np.random.default_rng(0)

    # Toy setup: 5 image-text 'aligned' pairs
    n_pairs = 5; d = 32
    img_encoder = make_encoder(n_pairs, d, rng)                  # each row = ideal image embed
    txt_encoder = make_encoder(n_pairs, d, rng)
    # Align them: text_j embed close to image_j (add small noise)
    txt_encoder = img_encoder + 0.1 * rng.normal(size=(n_pairs, d))

    # Normalise
    img_emb = np.array([normalize(v) for v in img_encoder])
    txt_emb = np.array([normalize(v) for v in txt_encoder])

    # Similarity matrix (batch)
    sim = img_emb @ txt_emb.T
    print(f"  Batch of {n_pairs} image-text pairs (dim = {d}):")
    print(f"  Sim matrix (rows=images, cols=texts):")
    for i in range(n_pairs):
        print(f"    {np.round(sim[i], 2).tolist()}")
    L = contrastive_loss(sim)
    print(f"  Contrastive loss (with aligned diagonal): {L:.4f}")
    # Randomised comparison
    sim_rand = np.random.default_rng(1).normal(scale=0.3, size=(n_pairs, n_pairs))
    print(f"  Contrastive loss (random pairing):        {contrastive_loss(sim_rand):.4f}")

    # Zero-shot classification
    print(f"\n  Zero-shot classification test:")
    for i in range(n_pairs):
        pred, scores = zero_shot_classify(img_emb[i], list(range(n_pairs)), txt_emb)
        print(f"    Image {i} -> class {pred}   {'✓' if pred == i else '✗'}")

    print("\n--- library cross-check (openai/CLIP; open_clip Python; sentence-transformers) ---")
