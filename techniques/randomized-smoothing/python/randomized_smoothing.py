"""Randomised smoothing / certified L2 robustness (Reference Ch 30 Robustness).

Cohen, Rosenfeld & Kolter (2019) "Certified Adversarial Robustness via
Randomised Smoothing."  Given any base classifier f, define the SMOOTHED
classifier g by majority vote over Gaussian noise:

  g(x) = argmax_c   P_{delta ~ N(0, sigma^2 I)} [ f(x + delta) = c ].

Cohen's Theorem: if the top class c has smoothed probability p_A and the
runner-up has p_B, then for any perturbation with |delta|_2 <= R, where

  R = sigma / 2 * ( Phi^-1(p_A) - Phi^-1(p_B) )

the smoothed classifier is guaranteed to still predict c. This is a
CERTIFIED (not empirical) L2-robustness radius.

In practice:
  - Sample n Monte-Carlo noise draws and estimate p_A, p_B by Clopper-
    Pearson lower / upper bounds at confidence 1 - alpha.
  - If the lower CI on p_A is <= 0.5, return ABSTAIN.
  - Else certify radius R using the estimated probabilities.

Here we demonstrate on a synthetic 2-class problem with a hand-tuned
base classifier; show the trade-off between sigma (bigger radius but
weaker base accuracy) and the certified radius.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays

from scipy.stats import norm as _norm, beta as _beta    # Gaussian + Beta cdfs


def base_classifier(x, centers):
    """Hard nearest-centre classifier (proxy for any pretrained f)."""
    dists = np.linalg.norm(x[:, None, :] - centers[None, :, :], axis=2)
    return dists.argmin(axis=1)


def smoothed_predict(x_batch, base_fn, sigma, n=1000, alpha=0.001, rng=None):
    """Certify (predicted_class, radius or NaN) for each input row."""
    if rng is None:
        rng = np.random.default_rng(0)
    n_in = x_batch.shape[0]
    d = x_batch.shape[1]
    out = np.full(n_in, -1, dtype=int)
    radius = np.full(n_in, np.nan)
    for i in range(n_in):
        x = x_batch[i]
        # Draw n noisy copies, classify.
        noise = rng.normal(0, sigma, (n, d))
        labels = base_fn(x[None, :] + noise)
        counts = np.bincount(labels)
        c_hat = counts.argmax()
        # Clopper-Pearson lower bound on p_A at level (1 - alpha).
        k = counts[c_hat]
        p_A_low = 0.0 if k == 0 else _beta.ppf(alpha, k, n - k + 1)
        if p_A_low <= 0.5:
            out[i] = -1                         # abstain
            continue
        # Cohen's theorem: R = sigma * Phi^-1(p_A_low)  (using p_B upper = 1 - p_A_low)
        r = sigma * _norm.ppf(p_A_low)
        out[i] = c_hat
        radius[i] = r
    return out, radius


if __name__ == "__main__":
    print("=== Randomised smoothing certified L2 robustness (Cohen 2019) ===\n")
    rng = np.random.default_rng(0)
    d = 8
    centers = np.array([np.zeros(d), np.eye(d)[0] * 4.0])   # two centres, distance 4 along axis 0

    n_test = 20
    y_true = rng.integers(0, 2, n_test)
    x_test = centers[y_true] + rng.normal(0, 0.3, (n_test, d))

    for sigma in (0.25, 0.5, 1.0):
        pred, rad = smoothed_predict(x_test, lambda X: base_classifier(X, centers),
                                       sigma=sigma, n=1000, rng=rng)
        n_certified = np.sum(~np.isnan(rad))
        n_abstain = np.sum(pred == -1)
        acc = np.mean(pred == y_true)
        mean_r = float(np.nanmean(rad)) if n_certified > 0 else float("nan")
        print(f"  sigma={sigma:.2f}   certified={n_certified}/{n_test}"
              f"   abstain={n_abstain}   accuracy={acc:.3f}   mean_R={mean_r:.3f}")

    print("\n  Larger sigma -> larger radius but more abstentions on ambiguous inputs.\n")
    print("--- library cross-check (smoothing-cohen ref repo; certified-robustness pip pkg) ---")
