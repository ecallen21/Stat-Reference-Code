"""Multivariate Normal (MVN) distribution — foundational.

X ~ N_p(mu, Sigma)
    E[X] = mu
    Cov[X] = Sigma
    Marginals, conditionals, linear combos all MVN.
    Quadratic form (X - mu)^T Sigma^{-1} (X - mu) ~ chi^2_p.

Sample via Cholesky:  X = mu + L Z,  Z ~ N(0, I),  L L^T = Sigma.
Log-density has closed form and enables MLE / Gaussian
processes / Kalman / etc.
"""

import numpy as np    # arrays + linalg


def sample_mvn(mu, Sigma, n, rng):
    L = np.linalg.cholesky(Sigma)
    Z = rng.standard_normal((n, len(mu)))
    return mu + Z @ L.T


def log_pdf_mvn(x, mu, Sigma):
    p = len(mu)
    d = x - mu
    sign, logdet = np.linalg.slogdet(Sigma)
    inv = np.linalg.solve(Sigma, d.T).T
    q = np.sum(d * inv, axis=1)
    return -0.5 * (p * np.log(2 * np.pi) + logdet + q)


def demo():
    print("=== Multivariate Normal (foundational) ===")
    rng = np.random.default_rng(2026)
    p = 3
    mu = np.array([1.0, -2.0, 3.0])
    Sigma = np.array([[2.0, 0.7, 0.3],
                      [0.7, 1.5, 0.4],
                      [0.3, 0.4, 1.0]])
    X = sample_mvn(mu, Sigma, n=10000, rng=rng)
    mu_hat = X.mean(axis=0)
    Sigma_hat = np.cov(X.T, bias=False)
    print(f"  MLE mu = {np.round(mu_hat, 3)}   (true {mu})")
    print(f"  MLE Sigma diagonal = {np.round(np.diag(Sigma_hat), 3)}  "
          f"(true {np.diag(Sigma)})")
    print(f"  MLE Sigma off-diag = {Sigma_hat[0, 1]:.3f}, {Sigma_hat[0, 2]:.3f}, "
          f"{Sigma_hat[1, 2]:.3f}  (true 0.7, 0.3, 0.4)")

    # Quadratic form => chi^2_p
    d = X - mu
    inv = np.linalg.solve(Sigma, d.T).T
    q = np.sum(d * inv, axis=1)
    print(f"\n  Mahalanobis-squared mean = {q.mean():.3f}, var = {q.var():.3f}")
    print(f"  Expected chi^2_{p}: mean = {p}, var = {2 * p}")

    print("\nSee also: gaussian-process-regression, kalman-filter (state-space),")
    print("          bayesian-linear-regression, probabilistic-pca, canonical-correlation,")
    print("          mahalanobis-distance-matching, hotellings-t2, manova.")


if __name__ == "__main__":
    demo()
