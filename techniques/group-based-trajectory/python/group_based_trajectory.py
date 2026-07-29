"""Group-Based Trajectory Modeling (Reference §12.5; also covers §12.6 GMM, §12.7 LCGA).

Assumes the population is a MIXTURE of K latent trajectory classes, each with
its own polynomial trajectory over time:

    y_{ij} | (subject i in class k)  ~  N(f_k(t_{ij}), sigma^2)
    f_k(t) = alpha_k0 + alpha_k1 * t + alpha_k2 * t^2 + ...

pi_k = P(subject i in class k)

Fit by EM:
    E-step   : posterior P(class k | subject i's whole trajectory)
    M-step   : update polynomial coefficients per class (weighted OLS) and pi_k

Variants:
    LCGA (§12.7)  : GBTM with sigma^2 fixed (no within-class variation)
    GMM (§12.6)   : GBTM + within-class random effects (mixture of LMMs)

This file implements GBTM (LCGA is the special case with sigma^2 free per class
but no random effects). BIC to pick K.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)


def _log_normal_pdf(y, mu, sigma):
    return -0.5 * math.log(2 * math.pi) - np.log(sigma) - 0.5 * ((y - mu) / sigma) ** 2


def fit_gbtm(y, time, subject_ids, K: int, degree: int = 2,
             max_iter: int = 200, tol: float = 1e-6, n_restarts: int = 5,
             seed: int = 0) -> dict:
    """EM fit for K-class group-based trajectory model with polynomial trajectories.

    Parameters
    ----------
    y : outcome per row, length n.
    time : time per row, length n.
    subject_ids : subject identifier per row.
    K : number of latent classes.
    degree : polynomial degree per class trajectory.
    """
    y = np.asarray(y, dtype=float); time = np.asarray(time, dtype=float)
    subject_ids = np.asarray(subject_ids)
    unique = np.unique(subject_ids)
    n_subj = len(unique)
    # Design matrix: intercept + t + t^2 + ...
    T_row = np.column_stack([time ** d for d in range(degree + 1)])
    p = T_row.shape[1]

    rng = np.random.default_rng(seed)
    best = None; best_ll = -np.inf
    for restart in range(n_restarts):
        # Random init of pi and class coefficient means
        pi = rng.dirichlet(np.ones(K))
        alpha = rng.normal(y.mean(), y.std(), size=(K, p))
        sigma = y.std()
        ll_prev = -np.inf
        for it in range(max_iter):
            # E-step: posterior class prob per subject
            log_r = np.zeros((n_subj, K))
            for s_idx, s in enumerate(unique):
                m = subject_ids == s
                y_i = y[m]; T_i = T_row[m]
                for k in range(K):
                    mu_ik = T_i @ alpha[k]
                    log_r[s_idx, k] = math.log(max(pi[k], 1e-12)) + _log_normal_pdf(y_i, mu_ik, sigma).sum()
            # normalize per row
            log_sum = np.max(log_r, axis=1, keepdims=True) + np.log(
                np.sum(np.exp(log_r - np.max(log_r, axis=1, keepdims=True)), axis=1, keepdims=True))
            gamma = np.exp(log_r - log_sum)                   # posteriors (n_subj x K)
            ll_marg = float(log_sum.sum())
            if abs(ll_marg - ll_prev) < tol: break
            ll_prev = ll_marg
            # M-step: pi
            pi = gamma.mean(axis=0)
            pi = np.clip(pi, 1e-6, 1.0); pi = pi / pi.sum()
            # M-step: alpha_k via weighted OLS
            for k in range(K):
                W_row = np.repeat(gamma[:, k], [np.sum(subject_ids == s) for s in unique])
                # solve (T'WT) alpha_k = T'W y with a tiny ridge to avoid
                # singular-matrix errors when a class collapses (near-empty)
                WT = T_row * W_row[:, None]
                A = WT.T @ T_row + 1e-8 * np.eye(T_row.shape[1])
                alpha[k] = np.linalg.solve(A, WT.T @ y)
            # sigma
            resid_sq = 0.0; total_w = 0.0
            for s_idx, s in enumerate(unique):
                m = subject_ids == s
                y_i = y[m]; T_i = T_row[m]
                for k in range(K):
                    mu_ik = T_i @ alpha[k]
                    resid_sq += gamma[s_idx, k] * ((y_i - mu_ik) ** 2).sum()
                    total_w += gamma[s_idx, k] * len(y_i)
            sigma = math.sqrt(resid_sq / total_w)
        if ll_prev > best_ll:
            best_ll = ll_prev
            best = {"pi": pi.copy(), "alpha": alpha.copy(), "sigma": float(sigma),
                     "gamma": gamma.copy(), "n_iter": it + 1}

    # BIC = -2 log L + p_free log(n_total)
    p_free = K * p + (K - 1) + 1                          # class coefs + pi + sigma
    n_total = len(y)
    bic = -2 * best_ll + p_free * math.log(n_total)
    aic = -2 * best_ll + 2 * p_free
    # Modal class assignment per subject
    modal = np.argmax(best["gamma"], axis=1)
    class_sizes = np.array([int((modal == k).sum()) for k in range(K)])
    return {"K": K, "degree": degree,
            "pi": best["pi"].tolist(),
            "alpha_per_class": best["alpha"].tolist(),
            "sigma": best["sigma"],
            "log_lik": best_ll,
            "BIC": bic, "AIC": aic,
            "modal_class_head": modal[:10].tolist(),
            "class_sizes_modal": class_sizes.tolist(),
            "n_iter": best["n_iter"], "n_subjects": int(n_subj),
            "method": "GBTM: EM K-class polynomial-trajectory mixture"}


def bic_select_k(y, time, subject_ids, k_grid=range(1, 5), degree: int = 2, seed: int = 0) -> dict:
    """Fit GBTM for a range of K; return the K with the smallest BIC."""
    results = []
    for K in k_grid:
        f = fit_gbtm(y, time, subject_ids, K, degree, seed=seed)
        results.append({"K": K, "log_lik": f["log_lik"], "BIC": f["BIC"], "AIC": f["AIC"]})
    best = min(results, key=lambda r: r["BIC"])
    return {"per_K": results, "best_K_by_BIC": best["K"]}


if __name__ == "__main__":
    rng = np.random.default_rng(31)
    n_subj = 100; n_time = 6
    subject_ids = np.repeat(np.arange(n_subj), n_time)
    time = np.tile(np.arange(n_time, dtype=float), n_subj)
    # 3 latent classes with different trajectories
    true_pi = [0.4, 0.35, 0.25]
    classes = rng.choice(3, size=n_subj, p=true_pi)
    trajectories = {
        0: lambda t: 5 + 0.5 * t,                # rising
        1: lambda t: 3 - 0.2 * t,                # slowly falling
        2: lambda t: 4 + 1.5 * t - 0.2 * t ** 2  # peak then decline
    }
    y = np.empty(n_subj * n_time)
    for i, s in enumerate(range(n_subj)):
        m = subject_ids == s
        y[m] = trajectories[classes[i]](time[m]) + rng.normal(0, 0.3, m.sum())

    print("=== GBTM at K=3 (true classes with pi = [0.4, 0.35, 0.25]) ===")
    fit = fit_gbtm(y, time, subject_ids, K=3, degree=2, n_restarts=5)
    print(f"  pi (est) = {[f'{p:.3f}' for p in fit['pi']]}")
    print(f"  sigma    = {fit['sigma']:.4f}")
    print(f"  log-lik  = {fit['log_lik']:.3f}")
    print(f"  BIC      = {fit['BIC']:.3f}")
    print(f"  modal-class sizes = {fit['class_sizes_modal']}")
    print(f"  class trajectories (intercept, linear, quad):")
    for k, coef in enumerate(fit['alpha_per_class']):
        print(f"    class {k}: {[f'{c:+.3f}' for c in coef]}")

    print("\n=== BIC selection K = 1..5 ===")
    sel = bic_select_k(y, time, subject_ids, range(1, 6), degree=2)
    for r in sel["per_K"]:
        print(f"  K = {r['K']}: log-lik = {r['log_lik']:.2f}, BIC = {r['BIC']:.2f}")
    print(f"  best K by BIC = {sel['best_K_by_BIC']}")
