"""Gradient clipping (Reference Ch 30 Robustness).

Pascanu, Mikolov & Bengio (2013) "On the difficulty of training recurrent
neural networks."  Two flavours:

  1. GLOBAL NORM CLIP (torch.nn.utils.clip_grad_norm_):
     if |g|_2 > tau:   g <- g * tau / |g|_2.
     Preserves direction; caps step magnitude.

  2. VALUE CLIP (torch.nn.utils.clip_grad_value_):
     g_i <- clip(g_i, -tau, tau).
     Coordinate-wise; may change direction.

Prevents exploding gradients (recurrent nets, deep transformers, mixed-
precision training, adversarial examples) and stabilises SGD in the
presence of heavy-tailed noise.

Here we demonstrate:
  (a) A synthetic scenario with a single OUTLIER gradient that would
      cause a big weight update; both clippers rescue the training run.
  (b) Impact on convergence for a small linear regression with heavy-
      tailed gradient noise -- unclipped diverges; norm-clip converges.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def clip_global_norm(g, tau):
    n = np.linalg.norm(g)
    if n > tau:
        return g * (tau / n), True
    return g, False


def clip_value(g, tau):
    return np.clip(g, -tau, tau), np.any(np.abs(g) > tau)


def train_lin_reg(x, y, tau=None, mode=None, lr=0.01, epochs=200,
                   heavy_tail_prob=0.05, heavy_tail_scale=100.0, seed=0):
    rng = np.random.default_rng(seed)
    d = x.shape[1]
    beta = np.zeros(d)
    n = x.shape[0]
    losses = []; clip_events = 0
    for ep in range(epochs):
        # Batch of size n; noisy MSE gradient.
        g = 2 * x.T @ (x @ beta - y) / n
        # Inject occasional huge gradient (simulates outlier / exploding grad).
        if rng.random() < heavy_tail_prob:
            g = g + heavy_tail_scale * rng.normal(0, 1, d)
        if mode == "norm":
            g, clipped = clip_global_norm(g, tau)
        elif mode == "value":
            g, clipped = clip_value(g, tau)
        else:
            clipped = False
        beta -= lr * g
        clip_events += int(clipped)
        losses.append(float(np.mean((x @ beta - y) ** 2)))
    return beta, losses, clip_events


if __name__ == "__main__":
    print("=== Gradient clipping (Pascanu 2013) ===\n")

    # (a) Single-shot demo: gradient with big spike.
    rng = np.random.default_rng(0)
    g = rng.normal(0, 1, 10)
    g_out = g.copy(); g_out[0] = 50.0                # outlier coordinate

    for tau in (1.0, 5.0):
        gn, hit_n = clip_global_norm(g_out, tau)
        gv, hit_v = clip_value(g_out, tau)
        print(f"  tau={tau:.1f}   |g|_2={np.linalg.norm(g_out):.2f}"
              f"   |g_norm|_2={np.linalg.norm(gn):.3f}  (clipped={hit_n})"
              f"   max|g_val|={np.max(np.abs(gv)):.3f}  (clipped={hit_v})")
    print()

    # (b) Linear regression convergence under occasional huge grads.
    rng = np.random.default_rng(1)
    d = 5
    beta_true = rng.normal(0, 1, d)
    X = rng.normal(0, 1, (200, d))
    y = X @ beta_true + rng.normal(0, 0.5, 200)

    print(f"  {'mode':>8s}   {'tau':>5}   {'final_beta_err':>14}   {'clip_events':>12}"
          f"   {'final_train_mse':>14}")
    for mode, tau in (("none  ", None), ("norm  ", 5.0), ("value ", 1.0)):
        beta, losses, clips = train_lin_reg(X, y, tau=tau, mode=None if mode.strip() == "none"
                                              else mode.strip())
        err = np.linalg.norm(beta - beta_true)
        print(f"  {mode:>8s}   {tau if tau is not None else '-':>5}   {err:>14.4f}"
              f"   {clips:>12}   {losses[-1]:>14.4f}")

    print("\n  Unclipped diverges on the heavy-tailed gradient noise.\n"
          "  Norm clipping keeps direction intact; value clipping also works but distorts direction.\n")
    print("--- library cross-check (torch.nn.utils.clip_grad_norm_ / clip_grad_value_) ---")
