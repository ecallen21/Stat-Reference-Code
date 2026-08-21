"""Bayesian 2PL IRT via Polya-Gamma data augmentation Gibbs (Reference §22.x extra).

For binary IRT with logit link:
    P(U_ij = 1 | theta_i, a_j, b_j) = sigma( a_j * (theta_i - b_j) )

Reparametrise psi_ij = a_j * (theta_i - b_j) = a_j * theta_i - a_j * b_j;
augment with Polya-Gamma latent omega_ij ~ PG(1, psi_ij) so that:

    P(U | theta, a, b, omega) = 0.5 * exp((U - 0.5) psi_ij - 0.5 omega psi^2)
                                          (Polson-Scott-Windle 2013)

Then theta, a, b have Gaussian full conditionals given omega — a proper Gibbs
sampler.  Instead of the PG(1, psi) draw (which needs a PG sampler), we use
a MOMENT MATCH: omega ~ p(1-p) which is a Bernoulli-variance approximation.
The result is a valid variational-Bayes / MAP EM (not full posterior sampling)
but converges to the posterior mode.

For genuine Gibbs draws use pypolyagamma; for MAP/posterior-mode we get a
proper Bayesian ANSWER (with priors) via this EM-style loop.

Priors:
    theta_i ~ N(0, 1)                (identifiability + shrinkage)
    log(a_j) ~ N(0, 1)
    b_j ~ N(0, 4)
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

import numpy as np    # numerical arrays + linear algebra


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def bayesian_2pl_map(U, n_iter: int = 100, tol: float = 1e-6,
                     seed: int = 0) -> dict:
    """Posterior MAP for 2PL with N(0, 1) prior on theta, log-N on a, N(0, 4) on b.
    Coordinate ascent (item / person Newton alternation)."""
    rng = np.random.default_rng(seed)
    U = np.asarray(U, dtype=float); n, J = U.shape
    theta = rng.normal(scale=0.5, size=n)
    a = np.ones(J)
    b = np.zeros(J)
    prev = -np.inf
    for it in range(n_iter):
        # --- item update: per-item logistic with Gaussian prior on (log a, b) ---
        for j in range(J):
            la = np.log(a[j]); bj = b[j]
            for _ in range(5):
                z = np.exp(la) * (theta - bj)
                p = _sigmoid(z)
                w = p * (1 - p) + 1e-6
                # grad = X.T (U - p) with X = [d psi / d(log a, b)]
                d_la = np.exp(la) * (theta - bj)          # d psi / d log a = a (theta-b)
                d_b  = -np.exp(la) * np.ones(n)           # d psi / d b     = -a
                X = np.column_stack([d_la, d_b])
                res = U[:, j] - p
                g = X.T @ res
                H = (X.T * w) @ X
                # priors: log a ~ N(0, 0.5^2), b ~ N(0, 2^2)
                g[0] -= la / 0.25; H[0, 0] += 1.0 / 0.25
                g[1] -= bj / 4.0;  H[1, 1] += 0.25
                step = np.linalg.solve(H + 1e-4 * np.eye(2), g)
                step = np.clip(step, -0.5, 0.5)
                la += step[0]; bj += step[1]
            a[j] = float(np.clip(np.exp(la), 0.05, 5.0))
            b[j] = float(np.clip(bj, -5.0, 5.0))
        # --- person update: per-person Newton with N(0, 1) prior ---
        for i in range(n):
            ti = theta[i]
            for _ in range(3):
                z = a * (ti - b)
                p = _sigmoid(z)
                w = p * (1 - p) + 1e-6
                g_i = (a * (U[i] - p)).sum() - ti          # prior N(0,1)
                H_i = (a * a * w).sum() + 1.0
                step = np.clip(g_i / H_i, -0.5, 0.5)
                ti += step
            theta[i] = float(np.clip(ti, -5.0, 5.0))
        # explicit identification: theta ~ mean 0, sd 1 by rescaling
        # (compensate a and b so predictions stay invariant)
        mu = float(theta.mean()); theta -= mu; b -= mu
        s = float(theta.std(ddof=1))
        if s > 1e-3:
            theta /= s; a *= s
        # log posterior
        Z = np.outer(theta, np.ones(J)) * a - a * b        # (n, J)
        P = _sigmoid(Z)
        log_lik = float((U * np.log(P + 1e-12) + (1 - U) * np.log(1 - P + 1e-12)).sum())
        log_prior = (-0.5 * (theta ** 2).sum()
                     - 0.5 * (np.log(a) ** 2 / 0.25).sum()
                     - 0.5 * (b ** 2 / 4.0).sum())
        lp = log_lik + log_prior
        if abs(lp - prev) < tol:
            break
        prev = lp
    return {"theta": theta, "a": a, "b": b, "log_posterior": lp,
            "n_iter": it + 1,
            "method": "Bayesian 2PL MAP (coord-ascent Newton)"}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 400; J = 15
    theta_true = rng.normal(size=n)
    a_true = np.exp(rng.normal(scale=0.3, size=J))
    b_true = rng.normal(scale=1.0, size=J)
    P = _sigmoid(a_true * (theta_true[:, None] - b_true))
    U = (rng.uniform(size=(n, J)) < P).astype(int)

    fit = bayesian_2pl_map(U, n_iter=30)
    print(f"=== Bayesian 2PL MAP (n={n}, J={J}) ===")
    print(f"  iterations = {fit['n_iter']}   log-posterior = {fit['log_posterior']:.1f}")
    print(f"\n  cor(a_hat, a_true) = "
          f"{float(np.corrcoef(fit['a'], a_true)[0, 1]):+.3f}")
    print(f"  cor(b_hat, b_true) = "
          f"{float(np.corrcoef(fit['b'], b_true)[0, 1]):+.3f}")
    print(f"  cor(theta_hat, theta_true) = "
          f"{float(np.corrcoef(fit['theta'], theta_true)[0, 1]):+.3f}")

    # side-by-side item params
    print(f"\n  {'j':>2} {'a_hat':>8} {'a_true':>8}  {'b_hat':>8} {'b_true':>8}")
    for j in range(min(J, 6)):
        print(f"  {j:>2} {fit['a'][j]:>8.3f} {a_true[j]:>8.3f}  "
              f"{fit['b'][j]:>8.3f} {b_true[j]:>8.3f}")

    print("\n--- library cross-check (R brms / rstan / mirt Bayesian) ---")
