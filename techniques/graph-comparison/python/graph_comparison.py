"""Graph comparison: spectral distance, feature-signature distance, DeltaCon similarity (Reference §24.12).

Given two graphs G1, G2 on the SAME node set, quantify their (dis)similarity:

  * Spectral distance: Euclidean distance between sorted eigenvalue vectors
    of the (normalized) Laplacian.
  * Feature-signature distance: differences in scalar summaries
    (density, mean degree, clustering, ...).
  * DeltaCon (Koutra et al. 2013): affinity via S = (I + eps^2 * D - eps * A)^{-1};
    similarity = 1 / (1 + rho(S1, S2)),  where rho is the Matusita rooted-sum-square distance:
        rho(S1, S2) = sqrt( sum_ij ( sqrt(S1_ij) - sqrt(S2_ij) )^2 )

Different-size graphs need graph kernels or graph-edit-distance approximations
(gm-tools, GMatch4py) — deferred to the R stub.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

import numpy as np    # numerical arrays + linear algebra


def spectral_distance(A1, A2, normalized: bool = True) -> float:
    def _laplacian_spectrum(A):
        A = np.asarray(A, dtype=float); n = A.shape[0]
        d = A.sum(axis=1)
        if normalized:
            d_is = 1.0 / np.sqrt(np.maximum(d, 1e-9))
            L = np.eye(n) - (A * d_is[:, None]) * d_is[None, :]
        else:
            L = np.diag(d) - A
        return np.sort(np.linalg.eigvalsh(L))
    s1 = _laplacian_spectrum(A1); s2 = _laplacian_spectrum(A2)
    # pad shorter with zeros if node counts differ
    n = max(len(s1), len(s2))
    p1 = np.concatenate([s1, np.zeros(n - len(s1))])
    p2 = np.concatenate([s2, np.zeros(n - len(s2))])
    return float(np.linalg.norm(p1 - p2))


def feature_signature(A) -> dict:
    A = np.asarray(A); n = A.shape[0]
    d = A.sum(axis=1)
    return {"density": float(A.sum() / (n * (n - 1))),
            "mean_degree": float(d.mean()),
            "max_degree": int(d.max()),
            "n_triangles": int(np.trace(np.linalg.matrix_power(A.astype(float), 3)) // 6),
            "degree_var": float(d.var(ddof=1))}


def feature_signature_distance(A1, A2, weights=None) -> float:
    f1 = feature_signature(A1); f2 = feature_signature(A2)
    keys = list(f1.keys())
    if weights is None:
        weights = {k: 1.0 for k in keys}
    return float(np.sqrt(sum(weights[k] * (f1[k] - f2[k]) ** 2 for k in keys)))


def deltacon_similarity(A1, A2, epsilon: float = None) -> float:
    A1 = np.asarray(A1, dtype=float); A2 = np.asarray(A2, dtype=float)
    n = A1.shape[0]; I = np.eye(n)
    if epsilon is None:
        max_deg = max(A1.sum(axis=1).max(), A2.sum(axis=1).max())
        epsilon = 1.0 / (1.0 + max_deg)
    D1 = np.diag(A1.sum(axis=1)); D2 = np.diag(A2.sum(axis=1))
    S1 = np.linalg.inv(I + epsilon ** 2 * D1 - epsilon * A1)
    S2 = np.linalg.inv(I + epsilon ** 2 * D2 - epsilon * A2)
    rho = float(np.sqrt(((np.sqrt(np.abs(S1)) - np.sqrt(np.abs(S2))) ** 2).sum()))
    return 1.0 / (1.0 + rho)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 40
    p = 0.15
    # base graph
    A = (rng.uniform(size=(n, n)) < p).astype(int)
    A = np.triu(A, 1); A = A + A.T
    # perturbations: 5%, 25%, 50% edges flipped
    def _perturb(A, frac, seed):
        rr = np.random.default_rng(seed); B = A.copy()
        for i in range(n):
            for j in range(i + 1, n):
                if rr.uniform() < frac:
                    B[i, j] = B[j, i] = 1 - B[i, j]
        return B
    A_5 = _perturb(A, 0.05, seed=1)
    A_25 = _perturb(A, 0.25, seed=2)
    A_50 = _perturb(A, 0.50, seed=3)

    print("=== Graph comparison: base vs perturbations ===")
    print(f"  {'perturb':>10}  {'spec_dist':>10}  {'feat_dist':>10}  {'DeltaCon sim':>12}")
    for name, B in [("self", A), ("5% flip", A_5),
                    ("25% flip", A_25), ("50% flip", A_50)]:
        sd = spectral_distance(A, B)
        fd = feature_signature_distance(A, B)
        dc = deltacon_similarity(A, B)
        print(f"  {name:>10}  {sd:>10.4f}  {fd:>10.4f}  {dc:>12.4f}")

    print("\n--- library cross-check (netrd / gmatch4py / graphkernels) ---")
