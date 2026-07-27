"""MANOVA: Multivariate Analysis of Variance (Reference §9.2).

Extends ANOVA from a scalar response to a p-dimensional response vector.
Given K groups and n_k p-dim observations in group k, test whether the group
mean vectors are equal.

Decomposition:
    T = H + E
    T = total sum-of-squares-and-cross-products (SSCP) matrix
    H = between-groups SSCP  (Hypothesis)
    E = within-groups SSCP   (Error)

Four common test statistics (functions of eigenvalues lambda_1..lambda_s of E^{-1} H):
    Wilks' Lambda      = prod_i 1 / (1 + lambda_i)      -- likelihood-ratio, most common
    Pillai's Trace     = sum_i lambda_i / (1 + lambda_i) -- most robust
    Hotelling-Lawley   = sum_i lambda_i                  -- generalization of Hotelling T^2
    Roy's Largest Root = max_i lambda_i                  -- most powerful under strict conditions

Each maps to an F-approximation for hypothesis testing. Reduce to 1-way case
(single categorical predictor); MANCOVA extension = adjust for covariates first,
then run MANOVA on residuals.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)
from typing import Sequence    # stdlib: type hint meaning 'indexable iterable' (list / tuple / array)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy import stats    # distributions, hypothesis tests, PPFs (norm, t, chi2, ttest_ind, ...)


def _sscp_matrices(groups):
    """Compute H, E, and T SSCP matrices for a list of n_k x p group matrices."""
    all_data = np.vstack(groups)
    N, p = all_data.shape
    grand_mean = all_data.mean(axis=0)
    T = np.zeros((p, p))
    H = np.zeros((p, p))
    E = np.zeros((p, p))
    for g in groups:
        n_k = g.shape[0]
        m_k = g.mean(axis=0)
        diff = (m_k - grand_mean)[:, None]
        H += n_k * (diff @ diff.T)
        cent = g - m_k
        E += cent.T @ cent
    T = H + E
    return H, E, T, N, p, len(groups)


def _eigen_e_inv_h(H, E):
    """Eigenvalues of E^{-1} H (returned in descending order)."""
    # Solve E * lambda = H via generalized eigenproblem
    try:
        M = np.linalg.solve(E, H)
    except np.linalg.LinAlgError:
        M = np.linalg.pinv(E) @ H
    ev = np.linalg.eigvals(M)
    ev = np.real_if_close(ev)
    ev = np.sort(np.real(ev))[::-1]
    return np.clip(ev, 0.0, None)


def manova(groups) -> dict:
    """Compute Wilks / Pillai / Hotelling-Lawley / Roy statistics + F approximations.

    ``groups`` : list of n_k x p numeric matrices (one per group).
    """
    H, E, T, N, p, K = _sscp_matrices(groups)
    lam = _eigen_e_inv_h(H, E)
    # degrees of freedom
    nu_h = K - 1                  # hypothesis df (# groups - 1)
    nu_e = N - K                  # error df
    s = min(p, nu_h)              # effective rank

    # ---- Wilks' Lambda
    wilks = float(np.prod(1.0 / (1.0 + lam)))
    # Rao's F approximation for Wilks
    # (there are a few variants; use the one that reduces to exact F when either p=1 or nu_h=1)
    t2 = (p ** 2 * nu_h ** 2 - 4) / (p ** 2 + nu_h ** 2 - 5)
    tt = math.sqrt(t2) if t2 > 0 else 1.0
    ms = nu_e + nu_h - (p + nu_h + 1) / 2
    y = wilks ** (1.0 / tt)
    F_wilks = ((1 - y) / y) * (ms * tt - p * nu_h / 2 + 1) / (p * nu_h)
    df1_w = p * nu_h
    df2_w = ms * tt - p * nu_h / 2 + 1
    p_wilks = float(stats.f.sf(F_wilks, df1_w, df2_w)) if df2_w > 0 else float("nan")

    # ---- Pillai's trace
    pillai = float((lam / (1.0 + lam)).sum())
    m_p = (abs(p - nu_h) - 1) / 2
    n_p = (nu_e - p - 1) / 2
    df1_p = s * (2 * m_p + s + 1)
    df2_p = s * (2 * n_p + s + 1)
    F_pillai = ((2 * n_p + s + 1) / (2 * m_p + s + 1)) * (pillai / (s - pillai)) if s - pillai > 0 else float("inf")
    p_pillai = float(stats.f.sf(F_pillai, df1_p, df2_p)) if df1_p > 0 and df2_p > 0 else float("nan")

    # ---- Hotelling-Lawley
    hl = float(lam.sum())
    df1_hl = s * (2 * m_p + s + 1)
    df2_hl = 2 * (s * n_p + 1)
    F_hl = (df2_hl / df1_hl) * hl / s if s > 0 else float("nan")
    p_hl = float(stats.f.sf(F_hl, df1_hl, df2_hl)) if df1_hl > 0 and df2_hl > 0 else float("nan")

    # ---- Roy's largest root
    roy = float(lam.max()) if lam.size else 0.0
    r = max(p, nu_h)
    df2_r = nu_e - r + nu_h
    F_roy = (df2_r / r) * roy if r > 0 and df2_r > 0 else float("nan")
    p_roy = float(stats.f.sf(F_roy, r, df2_r)) if r > 0 and df2_r > 0 else float("nan")

    return {"eigenvalues_of_E_inv_H": lam.tolist(),
            "Wilks":              {"stat": wilks, "F": F_wilks, "df1": df1_w, "df2": df2_w, "p": p_wilks},
            "Pillai":             {"stat": pillai, "F": F_pillai, "df1": df1_p, "df2": df2_p, "p": p_pillai},
            "Hotelling_Lawley":   {"stat": hl, "F": F_hl, "df1": df1_hl, "df2": df2_hl, "p": p_hl},
            "Roy_largest_root":   {"stat": roy, "F": F_roy, "df1": r, "df2": df2_r, "p": p_roy},
            "n_total": N, "p_dim": p, "K_groups": K,
            "method": "1-way MANOVA (Wilks / Pillai / Hotelling-Lawley / Roy)"}


def library_versions(groups):
    from statsmodels.multivariate.manova import MANOVA
    import pandas as pd
    Xc = np.vstack(groups)
    grp = np.concatenate([np.full(g.shape[0], i) for i, g in enumerate(groups)])
    df = pd.DataFrame(Xc, columns=[f"v{i}" for i in range(Xc.shape[1])])
    df["g"] = grp
    m = MANOVA.from_formula(f"{' + '.join(df.columns[:-1])} ~ C(g)", data=df)
    res = m.mv_test().results["C(g)"]["stat"]
    return {"statsmodels":
            {stat: {"F": float(res.loc[stat, "F Value"]),
                    "num_df": float(res.loc[stat, "Num DF"]),
                    "den_df": float(res.loc[stat, "Den DF"]),
                    "p": float(res.loc[stat, "Pr > F"])}
             for stat in ["Wilks' lambda", "Pillai's trace",
                          "Hotelling-Lawley trace", "Roy's greatest root"]}}


if __name__ == "__main__":
    rng = np.random.default_rng(17)
    p_dim = 3
    Sigma = np.array([[1.0, 0.3, 0.2],
                      [0.3, 1.0, 0.4],
                      [0.2, 0.4, 1.0]])
    g1 = rng.multivariate_normal([0.0, 0.0, 0.0], Sigma, 50)
    g2 = rng.multivariate_normal([0.5, 0.2, -0.3], Sigma, 55)
    g3 = rng.multivariate_normal([-0.3, 0.8, 0.4], Sigma, 60)
    print("=== 1-way MANOVA (3 groups, p = 3) ===")
    out = manova([g1, g2, g3])
    for key in ["Wilks", "Pillai", "Hotelling_Lawley", "Roy_largest_root"]:
        print(f"  {key:20s}: {out[key]}")
    print("  eigenvalues:", out["eigenvalues_of_E_inv_H"])

    print("\n--- library (statsmodels MANOVA) ---")
    lib = library_versions([g1, g2, g3])["statsmodels"]
    for k, v in lib.items():
        print(f"  {k}: {v}")
