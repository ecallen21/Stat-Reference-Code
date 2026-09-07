"""Score matching (Reference Sec 46.17).

Hyvarinen 2005 'Estimation of non-normalised statistical models by
score matching', JMLR. Estimates the SCORE of a model density
psi_model(x) = grad_x log p(x; theta) by minimising:

    J(theta) = (1/2) * E[|| psi_model(x) - psi_data(x) ||^2]

which via integration by parts becomes (Hyvarinen's identity):

    J(theta) = E[ (1/2) * ||psi(x)||^2 + sum_j d psi_j / d x_j ]

so the intractable log Z(theta) drops out. Fit theta to minimise
the empirical J on samples from p_data.

Related:
    * Sliced score matching  -- projections for high-dim
    * Denoising score matching  -- backbone of diffusion models
    * Ratio matching             -- Gutmann-Hyvarinen NCE cousin

We fit a 1-D unnormalised Gaussian log p(x) = -0.5 * a * (x-b)^2 + c
by score matching -- Hyvarinen's identity gives a closed form.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays
from scipy.optimize import minimize    # score matching


def score_matching_gaussian_loss(theta, X):
    """J = E[0.5 * (psi(x))^2 + d psi / d x] for psi(x) = -a * (x - b).

    d psi / d x = -a.
    """
    a, b = theta
    if a <= 0:
        return 1e12
    psi = -a * (X - b)
    return float(np.mean(0.5 * psi ** 2 + (-a)))


def fit_gaussian_score_matching(X):
    r = minimize(score_matching_gaussian_loss, x0=np.array([1.0, 0.0]),
                 args=(X,), method="Nelder-Mead")
    a, b = r.x
    #  Model: log p = -0.5 * a * (x - b)^2 + const  ->  N(b, 1/a)
    return {"mu": b, "sigma": 1 / np.sqrt(a), "a": a, "loss": r.fun}


if __name__ == "__main__":
    print("=== Score matching -- Hyvarinen 2005 ===\n")
    rng = np.random.default_rng(0)
    mu_true, sigma_true = 2.0, 1.5
    X = rng.normal(mu_true, sigma_true, size=1000)

    r = fit_gaussian_score_matching(X)
    print(f"  True  (mu, sigma) = ({mu_true:.3f}, {sigma_true:.3f})")
    print(f"  SM    (mu, sigma) = ({r['mu']:.3f}, {r['sigma']:.3f})")
    print(f"  MLE   (mu, sigma) = ({X.mean():.3f}, {X.std(ddof=1):.3f})")
    print(f"\n  Score matching recovers Gaussian parameters WITHOUT touching the")
    print(f"  normalising constant -- useful for models where log Z is intractable")
    print(f"  (energy-based models, unnormalised densities, deep score networks).")

    print("\n--- library cross-check (from-scratch in R and Python; sm4mb Python) ---")
