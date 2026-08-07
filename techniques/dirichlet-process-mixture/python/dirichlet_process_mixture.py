"""Dirichlet Process Gaussian Mixture Model (Reference §14.31).

Bayesian nonparametric mixture model where the number of components is
INFINITE a priori but only a finite (data-driven) number is used.

    G     ~ DP(alpha, G_0)                Dirichlet process prior
    theta_i ~ G
    y_i   ~ F(theta_i)                    Gaussian emission

Chinese Restaurant Process (CRP) representation
    Existing customer sits at table k with prob n_k / (n - 1 + alpha),
    or opens a new table with prob alpha / (n - 1 + alpha).

DP-Gaussian mixture Gibbs sampler (Neal 2000, Algorithm 3)
    For each i in random order:
        Marginalize theta out of the likelihood (conjugate Normal-Inv-Gamma
        base measure).
        Reassign i to existing table k with prob n_k * f(y_i | theta_k^-i)
        or new table with prob alpha * f(y_i | G_0).

Number of clusters concentrates around alpha * log(n) a priori.  Data
generally overrides the prior for moderate n.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy import stats    # SciPy statistical distributions (norm, t, chi2, f) and tests


def dp_gmm_gibbs(y, alpha: float = 1.0, mu_0: float = 0.0, kappa_0: float = 0.1,
                 a_0: float = 2.0, b_0: float = 1.0, n_iter: int = 300,
                 seed: int = 0) -> dict:
    """Collapsed Gibbs sampler for DP Gaussian mixture with Normal-Inv-Gamma base."""
    y = np.asarray(y, dtype=float); n = len(y)
    rng = np.random.default_rng(seed)
    z = np.zeros(n, dtype=int)  # start all in one cluster
    n_k = {0: n}
    # Cluster sufficient statistics: sum(y_j) and sum(y_j^2)
    S = {0: y.sum()}; SS = {0: (y ** 2).sum()}

    def marginal_pred_log(yi, n_k, S_k, SS_k):
        """Posterior-predictive log-density of a t distribution (Normal-InvGamma marginal)."""
        k = n_k
        kappa_n = kappa_0 + k
        mu_n = (kappa_0 * mu_0 + S_k) / kappa_n
        a_n = a_0 + k / 2
        b_n = b_0 + 0.5 * (SS_k - S_k ** 2 / k if k > 0 else 0) + \
              (kappa_0 * k / kappa_n) * ((S_k / max(k, 1) - mu_0) ** 2) / 2
        scale = b_n * (kappa_n + 1) / (a_n * kappa_n)
        return stats.t.logpdf(yi, df=2 * a_n, loc=mu_n, scale=math.sqrt(max(scale, 1e-8)))

    def prior_pred_log(yi):
        scale = b_0 * (kappa_0 + 1) / (a_0 * kappa_0)
        return stats.t.logpdf(yi, df=2 * a_0, loc=mu_0, scale=math.sqrt(scale))

    for it in range(n_iter):
        order = rng.permutation(n)
        for i in order:
            zi_old = z[i]
            n_k[zi_old] -= 1
            S[zi_old] -= y[i]; SS[zi_old] -= y[i] ** 2
            if n_k[zi_old] == 0:
                del n_k[zi_old]; del S[zi_old]; del SS[zi_old]
            # Compute log probs for existing clusters + new
            keys = list(n_k.keys())
            log_probs = np.array([math.log(n_k[k]) + marginal_pred_log(y[i], n_k[k], S[k], SS[k]) for k in keys])
            new_log = math.log(alpha) + prior_pred_log(y[i])
            log_probs = np.append(log_probs, new_log)
            log_probs -= log_probs.max()
            probs = np.exp(log_probs); probs /= probs.sum()
            choice = int(rng.choice(len(probs), p=probs))
            if choice == len(keys):
                new_id = max(list(n_k.keys()) + [-1]) + 1
                z[i] = new_id
                n_k[new_id] = 1; S[new_id] = y[i]; SS[new_id] = y[i] ** 2
            else:
                z[i] = keys[choice]
                n_k[z[i]] += 1; S[z[i]] += y[i]; SS[z[i]] += y[i] ** 2
    return {"z": z, "n_clusters": int(len(np.unique(z))),
            "cluster_sizes": {int(k): int(v) for k, v in n_k.items()},
            "method": "DP Gaussian mixture via CRP Gibbs (Neal 2000, Algorithm 3)"}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # True K = 3 well-separated Gaussians
    y = np.concatenate([rng.normal(-3, 0.6, 60),
                        rng.normal(0, 0.6, 90),
                        rng.normal(4, 0.6, 50)])
    true_labels = np.repeat([0, 1, 2], [60, 90, 50])

    print("=== DP Gaussian mixture, alpha = 1.0, n = 200 ===")
    r = dp_gmm_gibbs(y, alpha=1.0, n_iter=300, seed=1)
    print(f"  discovered clusters: {r['n_clusters']}")
    print(f"  cluster sizes:       {r['cluster_sizes']}")

    # Try to compare with truth: for each true class, find the majority discovered cluster
    from collections import Counter
    for k in range(3):
        c = Counter(r["z"][true_labels == k])
        print(f"  true class {k}: majority-mapped to discovered cluster {c.most_common(1)[0][0]}"
              f" ({c.most_common(1)[0][1]} of {int((true_labels == k).sum())})")

    print("\n=== Effect of alpha on number of clusters ===")
    for a in (0.1, 1.0, 5.0):
        r = dp_gmm_gibbs(y, alpha=a, n_iter=150, seed=2)
        print(f"  alpha = {a:4.1f}  ->  clusters = {r['n_clusters']}")
