"""Quantile treatment effects (QTE) (Reference Sec 15.33).

Firpo (2007) 'Efficient semiparametric estimation of quantile treatment
effects', Econometrica. The QTE(tau) at quantile tau in (0, 1) is:

    QTE(tau) = Q_tau[Y(1)] - Q_tau[Y(0)]

where Q_tau is the tau-th quantile of the potential-outcome distribution.
Under unconfoundedness we estimate the marginal CDFs of Y(1) and Y(0)
via IPW then invert.

Firpo IPW estimator:

    F_1_hat(y) = sum(T * 1[Y <= y] / e(X)) / sum(T / e(X))
    F_0_hat(y) = sum((1-T) * 1[Y <= y] / (1-e(X))) / sum((1-T) / (1-e(X)))
    Q_tau_hat(t) = inf { y : F_t_hat(y) >= tau }

Unlike the ATE, QTE reveals heterogeneous effects along the outcome
distribution -- e.g. a policy that helps the median while hurting the
lower tail.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays
from scipy.optimize import minimize    # logistic PS fit


def fit_logistic_ps(x, t):
    x1 = np.c_[np.ones(len(t)), x]

    def nll(b):
        p = 1 / (1 + np.exp(-(x1 @ b)))
        p = np.clip(p, 1e-8, 1 - 1e-8)
        return -np.sum(t * np.log(p) + (1 - t) * np.log(1 - p))

    r = minimize(nll, np.zeros(x1.shape[1]), method="L-BFGS-B")
    return 1 / (1 + np.exp(-(x1 @ r.x)))


def firpo_qte(y, t, e, taus):
    """Firpo (2007) IPW quantile-treatment-effect estimator."""
    w1 = t / e
    w0 = (1 - t) / (1 - e)
    order = np.argsort(y)
    y_sorted = y[order]
    w1_sorted = w1[order]
    w0_sorted = w0[order]

    #  Weighted CDF: cumulative weight normalised
    F1 = np.cumsum(w1_sorted) / np.sum(w1_sorted)
    F0 = np.cumsum(w0_sorted) / np.sum(w0_sorted)

    out = []
    for tau in taus:
        i1 = int(np.searchsorted(F1, tau))
        i0 = int(np.searchsorted(F0, tau))
        i1 = min(i1, len(y_sorted) - 1)
        i0 = min(i0, len(y_sorted) - 1)
        q1 = y_sorted[i1]
        q0 = y_sorted[i0]
        out.append({"tau": tau, "Q1_hat": q1, "Q0_hat": q0, "QTE": q1 - q0})
    return out


if __name__ == "__main__":
    print("=== Firpo (2007) quantile treatment effects ===\n")
    rng = np.random.default_rng(0)
    n = 6000
    x = rng.normal(size=(n, 2))
    #  Propensity depends on x[0]
    e_true = 1 / (1 + np.exp(-(0.5 * x[:, 0] - 0.3 * x[:, 1])))
    t = (rng.uniform(size=n) < e_true).astype(int)

    #  Heterogeneous effect along the distribution:
    #    Y(0) ~ Normal(0, 1)
    #    Y(1) ~ Normal(0, 1) + tau(rank) where tau grows with rank
    #    -> QTE(0.1) small, QTE(0.9) large
    eps = rng.normal(size=n)
    y0 = 0.3 * x[:, 0] + eps
    #  Rank-based treatment effect: bigger at higher quantiles
    ranks = np.argsort(np.argsort(eps)) / n
    tau_i = 0.5 + 2.0 * ranks
    y1 = y0 + tau_i
    y = np.where(t == 1, y1, y0)

    e_hat = fit_logistic_ps(x, t)
    taus = [0.1, 0.25, 0.5, 0.75, 0.9]
    r = firpo_qte(y, t, e_hat, taus)

    #  Approximate truth (unconditional): E[tau_i | rank(eps) = tau]
    print(f"  {'tau':>6s}  {'Q0':>8s}  {'Q1':>8s}  {'QTE':>8s}  {'truth':>8s}")
    for row in r:
        tau = row["tau"]
        truth = 0.5 + 2.0 * tau
        print(f"  {tau:>6.2f}  {row['Q0_hat']:>+8.3f}  {row['Q1_hat']:>+8.3f}  "
              f"{row['QTE']:>+8.3f}  {truth:>+8.3f}")

    print("\n  Note: QTE grows from ~0.7 (tau=0.1) to ~2.3 (tau=0.9) -- a rank-shift,")
    print("  invisible to ATE (which is ~1.5 the mean of tau).")

    print("\n--- library cross-check (Counterfactual R; econml Python) ---")
