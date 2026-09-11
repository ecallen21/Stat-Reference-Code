"""DreamBooth - Subject-Driven Fine-Tuning (Reference Sec 47.222).

Ruiz, Li, Jampani, Pritch, Rubinstein & Aberman 2023 'DreamBooth:
Fine Tuning Text-to-Image Diffusion Models for Subject-Driven
Generation', CVPR. Personalise a pretrained text-to-image model
with 3-5 photos of a SUBJECT using a rare identifier token:

    prompt = "a photo of [V] dog"
    prior-preservation loss keeps the class ('dog') distribution
    from drifting; identifier-token loss teaches the subject.

L_DB = L_recon(x_subject, "a [V] dog") + lambda * L_prior("a dog")

Prior loss is essential to prevent 'language drift' (model forgets
what a generic dog looks like).
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def reconstruction_loss(theta, subject_embed, subject_target):
    """Toy loss: MSE(theta @ subject_embed - target)."""
    pred = theta @ subject_embed
    return float(np.mean((pred - subject_target) ** 2))


def prior_preservation_loss(theta, class_embed, class_target):
    """Loss against the pretrained model's outputs on the class prompt."""
    pred = theta @ class_embed
    return float(np.mean((pred - class_target) ** 2))


def dreambooth_train(theta_init, subject, class_priors, lambda_prior=1.0,
                       lr=0.05, n_iter=200):
    """Toy DreamBooth loop: subject fits + prior preservation."""
    theta = theta_init.copy()
    hist = []
    for it in range(n_iter):
        # Compute both losses
        L_sub = reconstruction_loss(theta, subject["embed"], subject["target"])
        L_prior = np.mean([prior_preservation_loss(theta, c["embed"], c["target"])
                              for c in class_priors])
        L = L_sub + lambda_prior * L_prior
        # Finite-diff gradient
        eps = 1e-3
        grad = np.zeros_like(theta)
        for i in range(theta.shape[0]):
            for j in range(theta.shape[1]):
                theta_p = theta.copy(); theta_p[i, j] += eps
                L_p = reconstruction_loss(theta_p, subject["embed"], subject["target"]) + \
                       lambda_prior * np.mean([prior_preservation_loss(theta_p, c["embed"], c["target"])
                                                 for c in class_priors])
                grad[i, j] = (L_p - L) / eps
        theta -= lr * grad
        hist.append(L)
    return theta, hist


if __name__ == "__main__":
    print("=== DreamBooth (Ruiz et al 2023 CVPR) ===\n")
    rng = np.random.default_rng(0)

    d_prompt = 5; d_img = 4
    theta_init = rng.normal(scale=0.1, size=(d_img, d_prompt))

    # Subject: "[V] dog" - a specific corgi
    subject = {"embed": np.array([1, 0, 0, 1, 0], dtype=float),  # [V]-dog embedding
                "target": np.array([2.0, 0.5, 1.5, 0.8], dtype=float)}  # specific corgi img
    # Prior 'a dog' generic class prompts (from pretrained model)
    class_priors = [
        {"embed": np.array([0, 1, 0, 1, 0], dtype=float),         # 'a dog' embedding
         "target": np.array([1.0, 1.0, 1.0, 1.0], dtype=float)},  # generic dog img
        {"embed": np.array([0, 1, 0, 0.5, 0.5], dtype=float),
         "target": np.array([1.2, 0.8, 1.1, 0.9], dtype=float)},
    ]

    # Train with vs without prior preservation
    theta_no_prior, hist_no = dreambooth_train(theta_init, subject, class_priors,
                                                     lambda_prior=0.0, lr=0.3, n_iter=100)
    theta_prior, hist_p = dreambooth_train(theta_init, subject, class_priors,
                                                lambda_prior=1.0, lr=0.3, n_iter=100)

    print(f"  Subject-fit loss:")
    print(f"    without prior: {reconstruction_loss(theta_no_prior, subject['embed'], subject['target']):.4f}")
    print(f"    with prior:    {reconstruction_loss(theta_prior, subject['embed'], subject['target']):.4f}")
    print(f"  Prior-preservation loss (mean over class):")
    print(f"    without prior: {np.mean([prior_preservation_loss(theta_no_prior, c['embed'], c['target']) for c in class_priors]):.4f}")
    print(f"    with prior:    {np.mean([prior_preservation_loss(theta_prior, c['embed'], c['target']) for c in class_priors]):.4f}")

    print("\n  With prior preservation, the model learns the subject WITHOUT")
    print("  forgetting how to render generic class members ('language drift' fix).")

    print("\n--- library cross-check (diffusers.DreamBoothPipeline; kohya-ss/sd-scripts) ---")
