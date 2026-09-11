"""Classifier-Free Guidance - CFG (Reference Sec 47.210).

Ho & Salimans 2021 'Classifier-Free Diffusion Guidance', NeurIPS
Workshop. Instead of using an external classifier to steer
diffusion samples toward a class, jointly train the score model
CONDITIONAL and UNCONDITIONAL (drop the conditioning ~10% of
training). At sampling:

    eps_hat = eps_uncond + w * (eps_cond - eps_uncond)

w = 1: standard conditional; w > 1: overshoot toward conditioning
(higher fidelity, less diversity); w < 0: negative prompt.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def cfg_combine(eps_cond, eps_uncond, w):
    """Apply CFG: interpolate / extrapolate between uncond and cond scores."""
    return eps_uncond + w * (eps_cond - eps_uncond)


def sample_1d(mode, w, n=1000, rng=None):
    """Toy 1D setup: cond points at mode (-3 or +3), uncond at N(0, 2)."""
    if rng is None: rng = np.random.default_rng(0)
    x_uncond_targets = rng.normal(scale=2.0, size=n)             # 'unconditional' distribution
    x_cond_target = mode * np.ones(n)                            # conditional peak
    # Fake score model: at each x, we know the true score for each distribution.
    # Score is -grad log p(x). For N(0, sigma^2): score = -x / sigma^2.
    # Blend as if we started from x_T = 0 and take one Euler step of size 1
    x_uncond_score = -x_uncond_targets / (2.0 ** 2)
    x_cond_score = -(x_uncond_targets - mode) / (0.5 ** 2)
    eps = cfg_combine(x_cond_score, x_uncond_score, w)
    # Sample = uncond starting point + step in blended direction
    return x_uncond_targets + eps * 0.3


if __name__ == "__main__":
    print("=== Classifier-Free Guidance (Ho-Salimans 2021) ===\n")
    rng = np.random.default_rng(0)

    for w in [0.0, 1.0, 3.0, 7.5, 15.0]:
        samples = sample_1d(mode=3.0, w=w, n=5000, rng=rng)
        mean = float(np.mean(samples))
        std = float(np.std(samples))
        frac_near_mode = float(np.mean(np.abs(samples - 3.0) < 0.5))
        print(f"  w = {w:5.1f}   sample mean = {mean:6.3f}   std = {std:.3f}   "
              f"fraction within 0.5 of mode 3.0: {frac_near_mode:.3f}")

    print(f"\n  Higher w -> tighter concentration around the conditional target")
    print(f"  (higher fidelity, less diversity). Standard values: 5-10 for image gen.")

    # Negative prompt: w < 0 pushes away from the conditioning
    print(f"\n  Negative w:")
    for w in [-1.0, -3.0]:
        samples = sample_1d(mode=3.0, w=w, n=5000, rng=rng)
        frac_near_mode = float(np.mean(np.abs(samples - 3.0) < 0.5))
        frac_opposite = float(np.mean(samples < -1.0))
        print(f"  w = {w:5.1f}   fraction near mode 3.0 = {frac_near_mode:.3f}   "
              f"fraction pushed to negative side = {frac_opposite:.3f}")

    print("\n--- library cross-check (diffusers.StableDiffusionPipeline guidance_scale Python) ---")
