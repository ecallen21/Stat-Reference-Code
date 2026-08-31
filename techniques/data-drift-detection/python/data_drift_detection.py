"""Data-drift detection (Reference Ch 32 MLOps).

Detect changes in the INPUT distribution p(x) between a REFERENCE window
(e.g. training data) and a CURRENT window (production traffic).

Three standard scores:

  1. POPULATION STABILITY INDEX (PSI):
       Bin the feature; compare bin proportions.
         PSI = sum_b (q_b - p_b) * log(q_b / p_b).
         PSI < 0.1  -> no drift.  0.1 <= PSI < 0.25 -> moderate drift.
         PSI >= 0.25 -> severe drift.

  2. KOLMOGOROV-SMIRNOV (KS) two-sample statistic:
       max |F_ref(x) - F_cur(x)|. Univariate; distribution-free.

  3. WASSERSTEIN-1 (Earth-mover):
       int |F_ref^-1(u) - F_cur^-1(u)| du. Distance in the feature's
       original units; interpretable per-feature.

For multivariate data, monitor each feature independently PLUS a
MULTIVARIATE drift score via MMD or a domain-classifier (Rabanser 2019).

Here we implement PSI + KS + Wasserstein per feature on synthetic
training vs shifted-production data, and flag drift under the standard
thresholds.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def psi(reference, current, n_bins=10, epsilon=1e-4):
    """Population Stability Index (Wu-Olson 2010)."""
    edges = np.quantile(reference, np.linspace(0, 1, n_bins + 1))
    edges[0] = -np.inf; edges[-1] = np.inf
    p_ref, _ = np.histogram(reference, bins=edges)
    p_cur, _ = np.histogram(current, bins=edges)
    p = p_ref / max(p_ref.sum(), 1) + epsilon
    q = p_cur / max(p_cur.sum(), 1) + epsilon
    return float(np.sum((q - p) * np.log(q / p)))


def ks_statistic(reference, current):
    """Two-sample KS statistic."""
    all_x = np.concatenate([reference, current])
    all_x.sort()
    F_ref = np.searchsorted(np.sort(reference), all_x, side="right") / len(reference)
    F_cur = np.searchsorted(np.sort(current),   all_x, side="right") / len(current)
    return float(np.max(np.abs(F_ref - F_cur)))


def wasserstein1(reference, current):
    """Wasserstein-1 distance for 1-D data (sort + trapezoid on CDF gap)."""
    r = np.sort(reference); c = np.sort(current)
    # Interpolate onto a common grid via quantiles
    u = np.linspace(0, 1, max(len(r), len(c)))
    q_r = np.quantile(r, u); q_c = np.quantile(c, u)
    return float(np.mean(np.abs(q_r - q_c)))


def per_feature_report(ref_X, cur_X, feature_names=None, psi_flag=0.10):
    d = ref_X.shape[1]
    if feature_names is None:
        feature_names = [f"x{i}" for i in range(d)]
    rows = []
    for j in range(d):
        r_j, c_j = ref_X[:, j], cur_X[:, j]
        p = psi(r_j, c_j)
        k = ks_statistic(r_j, c_j)
        w = wasserstein1(r_j, c_j)
        status = "OK" if p < psi_flag else ("moderate" if p < 0.25 else "SEVERE")
        rows.append((feature_names[j], p, k, w, status))
    return rows


if __name__ == "__main__":
    print("=== Data-drift detection: PSI + KS + Wasserstein ===\n")
    rng = np.random.default_rng(0)
    n = 2000
    # Reference: 3 features from three different Gaussians.
    ref_X = np.stack([rng.normal(0, 1, n),
                       rng.normal(0, 1, n),
                       rng.normal(0, 1, n)], axis=1)
    # Production: feature 0 unchanged; feature 1 shifted by 0.4; feature 2 variance x 2.
    cur_X = np.stack([rng.normal(0, 1, n),
                       rng.normal(0.4, 1, n),
                       rng.normal(0, 2, n)], axis=1)

    print(f"  {'feature':>10}   {'PSI':>7}   {'KS':>6}   {'W1':>6}   {'status':>8}")
    for name, p, k, w, status in per_feature_report(ref_X, cur_X):
        print(f"  {name:>10}   {p:>7.3f}   {k:>6.3f}   {w:>6.3f}   {status:>8}")

    print("\n  Thresholds:  PSI < 0.10 OK   0.10 <= PSI < 0.25 moderate   >= 0.25 SEVERE.\n")
    print("--- library cross-check (evidently, alibi-detect, whylogs, nannyML) ---")
