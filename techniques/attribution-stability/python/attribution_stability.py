"""Feature-attribution stability (Reference Sec 47.52).

Yeh et al 2019 'On the (in)fidelity and sensitivity of explanations',
NeurIPS; Alvarez-Melis & Jaakkola 2018 'On the robustness of
interpretability methods'. Stability = how much a feature-attribution
method's output changes under (a) small input perturbations,
(b) model retraining on bootstrap resamples.

    Local Lipschitz:   L(x) = max_{x' in B(x, eps)} || phi(x) - phi(x') ||
                                / || x - x' ||

    Bootstrap Jaccard: J = |topk(A) ∩ topk(B)| / |topk(A) ∪ topk(B)|

Higher L = more sensitive (less stable); lower J = less agreement
across bootstrap models.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays
from sklearn.ensemble import GradientBoostingRegressor    # base model
from sklearn.inspection import permutation_importance    # attribution method


def local_lipschitz(x, attributions, model, X_ref, n_perturb=25, eps=0.1, rng=None):
    """Approximate max ||phi(x) - phi(x')|| / ||x - x'|| in eps-ball around x."""
    rng = rng or np.random.default_rng(0)
    lips = []
    for _ in range(n_perturb):
        delta = rng.normal(size=x.shape) * eps
        xp = x + delta
        # attribution at xp: use a local numerical gradient of predicted y
        base = model.predict(xp.reshape(1, -1))[0]
        grad = np.zeros_like(xp)
        h = 1e-3
        for j in range(len(xp)):
            xp[j] += h
            grad[j] = (model.predict(xp.reshape(1, -1))[0] - base) / h
            xp[j] -= h
        d_attr = np.linalg.norm(attributions - grad)
        d_x = np.linalg.norm(delta) + 1e-9
        lips.append(d_attr / d_x)
    return float(np.max(lips))


def bootstrap_topk_jaccard(X, y, k_top=3, n_boot=20, seed=0):
    """Fit models on bootstrap resamples; measure top-k feature set agreement."""
    rng = np.random.default_rng(seed)
    n = len(X)
    tops = []
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        m = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=b)
        m.fit(X[idx], y[idx])
        imp = permutation_importance(m, X, y, n_repeats=5, random_state=b).importances_mean
        tops.append(set(np.argsort(imp)[::-1][:k_top].tolist()))
    # pairwise mean Jaccard
    js = []
    for i in range(n_boot):
        for j in range(i + 1, n_boot):
            js.append(len(tops[i] & tops[j]) / len(tops[i] | tops[j]))
    return {"jaccard_mean": float(np.mean(js)), "jaccard_std": float(np.std(js))}


if __name__ == "__main__":
    print("=== Attribution stability (Yeh et al 2019; Alvarez-Melis 2018) ===\n")
    rng = np.random.default_rng(0)
    n, p = 400, 8
    X = rng.normal(size=(n, p))
    y = 2.0 * X[:, 0] - 1.5 * X[:, 1] + 0.8 * X[:, 2] + 0.3 * rng.normal(size=n)

    # Baseline vs weakly-identified: dims 3-7 are near-copies of dims 0-2
    X_corr = X.copy()
    X_corr[:, 3] = X_corr[:, 0] + 0.15 * rng.normal(size=n)
    X_corr[:, 4] = X_corr[:, 1] + 0.15 * rng.normal(size=n)
    X_corr[:, 5] = X_corr[:, 2] + 0.15 * rng.normal(size=n)
    X_corr[:, 6] = X_corr[:, 0] + 0.15 * rng.normal(size=n)
    X_corr[:, 7] = X_corr[:, 1] + 0.15 * rng.normal(size=n)

    for name, Xd in [("independent features", X), ("correlated features", X_corr)]:
        m = GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=0).fit(Xd, y)
        imp = permutation_importance(m, Xd, y, n_repeats=10, random_state=0).importances_mean
        L = local_lipschitz(Xd[0], imp, m, Xd, n_perturb=25, eps=0.05)
        J = bootstrap_topk_jaccard(Xd, y, k_top=3, n_boot=10, seed=0)
        print(f"  {name:22s} L(local Lipschitz)={L:6.2f}   "
              f"bootstrap top-3 Jaccard = {J['jaccard_mean']:.3f} +/- {J['jaccard_std']:.3f}")

    print("\n  Correlated features -> attributions swap across resamples -> lower Jaccard.")
    print("  This is a stability warning, NOT a bug in the attribution method.")
    print("\n--- library cross-check (iml stability R; captum / shap Python) ---")
