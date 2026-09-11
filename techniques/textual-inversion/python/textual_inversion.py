"""Textual Inversion (Reference Sec 47.223).

Gal, Alaluf, Atzmon, Patashnik, Bermano, Chechik & Cohen-Or 2022
'An Image is Worth One Word: Personalizing Text-to-Image
Generation using Textual Inversion', ICLR 2023. Learn a NEW
EMBEDDING vector v_* for a placeholder token [S*] such that the
FROZEN diffusion model reconstructs 3-5 subject photos when
prompted with [S*].

    theta (diffusion model)  = frozen
    v_star (single token embedding, dim ~ 768) = trainable
    L = ||eps_theta(x_subject, "A photo of [S*]", v_star) - eps||^2

~1000x cheaper than DreamBooth (only one embedding vector) but
less faithful for complex subjects.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def frozen_model(v_star, other_tokens, W_out):
    """Toy 'frozen' text-to-image: concatenate embeddings and linear-project."""
    tokens = np.concatenate([v_star, other_tokens.flatten()])
    return tokens @ W_out


def train_textual_inversion(subject_embeds, subject_targets, other_tokens_per_ex,
                              W_out, dim=8, lr=0.1, n_iter=500, seed=0):
    """Learn only v_star; W_out is frozen."""
    rng = np.random.default_rng(seed)
    v_star = rng.normal(scale=0.1, size=dim)
    losses = []
    for it in range(n_iter):
        # Sum loss across all subject examples
        grad = np.zeros(dim)
        L_total = 0
        for i in range(len(subject_targets)):
            other = other_tokens_per_ex[i]
            pred = frozen_model(v_star, other, W_out)
            err = pred - subject_targets[i]
            L_total += np.mean(err ** 2)
            # Gradient wrt v_star
            grad += (2 * W_out[:dim] @ err) / len(subject_targets)
        v_star -= lr * grad / len(subject_targets)
        losses.append(L_total / len(subject_targets))
    return v_star, losses


if __name__ == "__main__":
    print("=== Textual Inversion (Gal et al 2022) ===\n")
    rng = np.random.default_rng(0)

    d = 8; n_examples = 4
    other_dim = 3 * d                                            # 3 other tokens
    out_dim = 16
    W_out = rng.normal(scale=0.3, size=(d + other_dim, out_dim))  # frozen "model"

    # Simulate 4 subject photos with different "context" tokens
    subject_targets = []
    other_tokens_per_ex = []
    v_true = rng.normal(scale=0.5, size=d)                       # the ideal subject embedding
    for i in range(n_examples):
        other = rng.normal(size=(3, d))
        target = frozen_model(v_true, other, W_out) + rng.normal(scale=0.05, size=out_dim)
        subject_targets.append(target)
        other_tokens_per_ex.append(other)

    v_learned, losses = train_textual_inversion(
        [None] * n_examples, subject_targets, other_tokens_per_ex, W_out,
        dim=d, lr=0.05, n_iter=500, seed=0)

    print(f"  Learned {d}-dim token embedding from {n_examples} subject photos.")
    print(f"  Loss trace: init = {losses[0]:.4f} -> final = {losses[-1]:.4f}")
    print(f"  ||v_learned - v_true|| = {float(np.linalg.norm(v_learned - v_true)):.4f}")
    print(f"  Cosine(v_learned, v_true) = "
          f"{float(v_learned @ v_true / (np.linalg.norm(v_learned) * np.linalg.norm(v_true))):.4f}")

    # Compare parameter counts
    dreambooth_params = W_out.size                               # full model fine-tune
    ti_params = d
    print(f"\n  Parameters trained:")
    print(f"    DreamBooth (full-model fine-tune): {dreambooth_params:>7,}")
    print(f"    Textual Inversion (one embedding): {ti_params:>7,}   ({dreambooth_params // ti_params}x cheaper)")

    print("\n--- library cross-check (diffusers.StableDiffusionPipeline + textual-inversion training script) ---")
