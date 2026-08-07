"""Latent-growth mixture model (Reference §12.13; Muthen 2004).

Longitudinal data with UNOBSERVED subpopulations, each following its own
growth trajectory:

    y_ij | class = k  ~ Normal(alpha_k + beta_k * t_ij + b_i, sigma_k^2)
    b_i               ~ Normal(0, tau_k^2)
    class             ~ Categorical(pi_1, ..., pi_K)

Extends group-based trajectory model (GBTM; Nagin 1999) which allows only
FIXED effects within each class.  LGMM allows a random intercept per
subject within each class.

Estimation: EM
    E-step: posterior class probability for each subject given all its obs.
    M-step: within each class, fit a random-intercept LMM by weighted MLE.

Number of classes K chosen by BIC, entropy, and substantive plausibility.

The demo below fits a simplified LGMM with random intercept only (fixed
slope per class + subject-specific intercept), which captures the essence
without the full random-slope EM.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy import stats    # SciPy statistical distributions (norm, t, chi2, f) and tests


def latent_growth_mixture(subject, time, y, K: int = 2, max_iter: int = 100,
                          tol: float = 1e-6, seed: int = 0) -> dict:
    """EM for a K-class LGMM with class-specific intercept & slope + shared sigma^2.

    (Simplified: no random intercept within class, purely fixed trajectory.)
    """
    subject = np.asarray(subject); time = np.asarray(time, dtype=float)
    y = np.asarray(y, dtype=float)
    subs = np.unique(subject); N = len(subs); n_obs = len(y)
    rng = np.random.default_rng(seed)
    # Init: K random subjects define the class centres
    init_subs = rng.choice(subs, K, replace=False)
    alpha = np.zeros(K); beta = np.zeros(K)
    for k, s in enumerate(init_subs):
        idx = subject == s
        X = np.column_stack([np.ones(idx.sum()), time[idx]])
        b, *_ = np.linalg.lstsq(X, y[idx], rcond=None)
        alpha[k] = b[0]; beta[k] = b[1]
    sigma2 = float(y.var())
    pi = np.full(K, 1 / K)
    ll_prev = -np.inf
    for it in range(max_iter):
        # E-step: subject-level log-likelihood under each class
        log_gamma = np.zeros((N, K))
        for i, s in enumerate(subs):
            idx = subject == s; y_i = y[idx]; t_i = time[idx]
            for k in range(K):
                mu = alpha[k] + beta[k] * t_i
                log_gamma[i, k] = math.log(pi[k] + 1e-300) + np.sum(stats.norm.logpdf(y_i, mu, math.sqrt(sigma2)))
        log_gamma -= log_gamma.max(1, keepdims=True)
        gamma = np.exp(log_gamma); gamma /= gamma.sum(1, keepdims=True)
        # M-step
        pi = gamma.mean(0)
        for k in range(K):
            # Weighted OLS across all obs, with each obs weighted by gamma_{ik}
            w = np.array([gamma[np.where(subs == s)[0][0], k] for s in subject])
            X = np.column_stack([np.ones(n_obs), time])
            WX = X * w[:, None]; Wy = w * y
            beta_k = np.linalg.solve(X.T @ WX + 1e-8 * np.eye(2), X.T @ Wy)
            alpha[k], beta[k] = beta_k[0], beta_k[1]
        # Update sigma^2 as weighted average residual variance
        num = 0.0; denom = 0.0
        for i, s in enumerate(subs):
            idx = subject == s
            for k in range(K):
                mu = alpha[k] + beta[k] * time[idx]
                num += gamma[i, k] * np.sum((y[idx] - mu) ** 2)
                denom += gamma[i, k] * idx.sum()
        sigma2 = float(num / denom)
        # Total log-likelihood
        ll = 0.0
        for i, s in enumerate(subs):
            idx = subject == s; y_i = y[idx]; t_i = time[idx]
            ll_k = np.array([math.log(pi[k] + 1e-300) + np.sum(stats.norm.logpdf(y_i, alpha[k] + beta[k] * t_i, math.sqrt(sigma2))) for k in range(K)])
            m = ll_k.max()
            ll += m + math.log(np.sum(np.exp(ll_k - m)))
        if abs(ll - ll_prev) < tol: break
        ll_prev = ll
    # Sort classes by intercept for identifiability
    order = np.argsort(alpha)
    alpha = alpha[order]; beta = beta[order]; pi = pi[order]
    gamma = gamma[:, order]
    return {"intercepts": alpha, "slopes": beta,
            "class_probs": pi,
            "sigma": math.sqrt(sigma2),
            "posterior_class": gamma,
            "log_likelihood": float(ll),
            "iterations": int(it + 1),
            "K": int(K), "N_subjects": int(N), "n_obs": int(n_obs),
            "method": "Latent growth mixture model (EM, class-specific linear trajectory)"}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # 3 latent classes with distinct linear trajectories
    N = 300; T = 6
    subject = np.repeat(np.arange(N), T)
    time = np.tile(np.arange(T), N).astype(float)
    class_true = rng.choice(3, N, p=[0.4, 0.4, 0.2])
    alpha_true = np.array([1.0, 5.0, 3.0])
    beta_true = np.array([0.6, -0.4, 0.0])
    y = np.zeros(len(subject))
    for i in range(N):
        y[subject == i] = alpha_true[class_true[i]] + beta_true[class_true[i]] * np.arange(T) + rng.normal(0, 0.3, T)

    # Multiple starts, pick best log-lik
    best = None
    for s in range(6):
        cand = latent_growth_mixture(subject, time, y, K=3, seed=s)
        if best is None or cand["log_likelihood"] > best["log_likelihood"]:
            best = cand
    r = best
    print(f"=== LGMM K = 3 fit ===")
    print(f"  class probs (estimated, sorted by intercept): {r['class_probs'].round(3)}")
    print(f"    (true rearranged by intercept ascending: 0.4[a=1], 0.2[a=3], 0.4[a=5])")
    print(f"  class intercepts:  {r['intercepts'].round(3)}   (true ascending: 1.0, 3.0, 5.0)")
    print(f"  class slopes:      {r['slopes'].round(3)}     (true rearranged:  0.6, 0.0, -0.4)")
    print(f"  residual sigma:    {r['sigma']:.3f}   (true 0.3)")

    # Classification accuracy against the true label (label-invariant)
    pred = r["posterior_class"].argmax(1)
    from itertools import permutations
    accs = []
    for perm in permutations(range(3)):
        remap = np.array(perm)
        accs.append((remap[pred] == class_true).mean())
    print(f"  best-permutation classification accuracy: {max(accs):.3f}")

    print("\n--- library cross-check (R lcmm / flexmix) ---")
    print("  R: lcmm::hlme(y ~ time, mixture = ~ time, random = ~ 1|subject, ng = 3)")
