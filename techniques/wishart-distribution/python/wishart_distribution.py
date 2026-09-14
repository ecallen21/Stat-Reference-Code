"""Wishart distribution (Wishart 1928).

Distribution of scatter matrix S = X.T X for X ~ N_n(0, Sigma).
For n >= p:  S ~ Wishart_p(n, Sigma).

Conjugate prior for precision matrix in multivariate Normal.
Inverse-Wishart is the corresponding prior for the covariance.

E[S]  = n * Sigma
Var(S_ij) = n * (Sigma_ii Sigma_jj + Sigma_ij^2)
"""

import numpy as np    # arrays + linalg


def sample_wishart(n, Sigma, size, rng):
    """Bartlett decomposition for efficient sampling."""
    p = Sigma.shape[0]
    L = np.linalg.cholesky(Sigma)
    out = np.empty((size, p, p))
    for k in range(size):
        A = np.zeros((p, p))
        for i in range(p):
            A[i, i] = np.sqrt(rng.chisquare(n - i))
            for j in range(i):
                A[i, j] = rng.standard_normal()
        LA = L @ A
        out[k] = LA @ LA.T
    return out


def demo():
    print("=== Wishart distribution (Wishart 1928) ===")
    rng = np.random.default_rng(2026)
    p = 3
    n = 20
    Sigma = np.array([[2.0, 0.7, 0.3],
                      [0.7, 1.5, 0.4],
                      [0.3, 0.4, 1.0]])
    samples = sample_wishart(n, Sigma, size=5000, rng=rng)
    emp_mean = samples.mean(axis=0)
    tru_mean = n * Sigma
    err = np.linalg.norm(emp_mean - tru_mean) / np.linalg.norm(tru_mean)
    print(f"  p={p}, dof={n}")
    print(f"  Empirical E[S] Frobenius rel err vs n*Sigma = {err:.4f}")
    for i, j in [(0, 0), (0, 1), (1, 1), (2, 2)]:
        emp = emp_mean[i, j]
        tru = tru_mean[i, j]
        print(f"    S[{i}, {j}]: empirical = {emp:.3f}, theory = {tru:.3f}")

    print("\nSee also: covariance-estimation-highdim, bayesian-hierarchical-models,")
    print("          bayesian-glms, gaussian-graphical-model,")
    print("          mcmc-metropolis-hastings (conjugate updates in Bayesian workflows).")


if __name__ == "__main__":
    demo()
