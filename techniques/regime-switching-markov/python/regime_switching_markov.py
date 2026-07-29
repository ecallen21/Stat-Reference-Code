"""Markov-switching model (Hamilton 1989) - Reference §13.14 / §13.15.

Two-regime (or K-regime) model where the parameters of a time series depend
on a HIDDEN Markov state S_t in {1, ..., K}:

    y_t | S_t = k  ~  N(mu_k, sigma_k^2)                 (regime-dependent mean and variance)
    Pr(S_t = j | S_{t-1} = i) = P[i, j]                  (regime transition matrix)

Common in macro/finance: distinguish "expansion vs recession" (mu +, mu -)
or "low-volatility vs high-volatility" (sigma small, sigma large) regimes.
Related: Threshold AR (TAR), Smooth-Transition AR (STAR), regime-switching
GARCH.  Regime-switching GARCH (SWARCH) is Hamilton & Susmel (1994).

Estimation
    EM (Baum-Welch) on a hidden Markov model with Gaussian emissions:
        E-step: forward-backward filter/smoother -> gamma[t, k] = Pr(S_t = k | y_{1:T}),
                                                    xi[t, i, j] = Pr(S_t=i, S_{t+1}=j | y_{1:T})
        M-step: mu_k = sum_t gamma[t,k] y_t / sum_t gamma[t,k]
                sigma_k^2 = weighted variance
                P[i,j] = sum_t xi[t,i,j] / sum_t gamma[t,i]

Extension to Markov-switching AR(p) is analogous with regression on lagged y.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy import stats    # SciPy statistical distributions (norm, t, chi2, f) and tests


def _forward_backward(log_emis, log_pi, log_P):
    T, K = log_emis.shape
    log_alpha = np.full((T, K), -np.inf)
    log_alpha[0] = log_pi + log_emis[0]
    for t in range(1, T):
        for j in range(K):
            log_alpha[t, j] = log_emis[t, j] + np.max(log_alpha[t - 1] + log_P[:, j]) + \
                math.log(np.sum(np.exp(log_alpha[t - 1] + log_P[:, j] -
                                       np.max(log_alpha[t - 1] + log_P[:, j]))))
    log_beta = np.zeros((T, K))
    for t in range(T - 2, -1, -1):
        for i in range(K):
            v = log_P[i] + log_emis[t + 1] + log_beta[t + 1]
            log_beta[t, i] = np.max(v) + math.log(np.sum(np.exp(v - np.max(v))))
    log_gamma = log_alpha + log_beta
    log_gamma -= np.array([np.max(log_gamma, 1) + np.log(np.sum(np.exp(log_gamma - np.max(log_gamma, 1, keepdims=True)), 1))]).T
    gamma = np.exp(log_gamma)
    ll = float(np.max(log_alpha[-1]) + math.log(np.sum(np.exp(log_alpha[-1] - np.max(log_alpha[-1])))))
    xi = np.zeros((T - 1, K, K))
    for t in range(T - 1):
        m = np.full((K, K), -np.inf)
        for i in range(K):
            for j in range(K):
                m[i, j] = log_alpha[t, i] + log_P[i, j] + log_emis[t + 1, j] + log_beta[t + 1, j]
        m -= np.max(m)
        m_exp = np.exp(m); xi[t] = m_exp / m_exp.sum()
    return gamma, xi, ll


def markov_switching_normal(y, K: int = 2, max_iter: int = 200, tol: float = 1e-6,
                            seed: int = 0) -> dict:
    """Fit K-regime Markov-switching Gaussian mean/variance model by EM."""
    y = np.asarray(y, dtype=float); T = len(y)
    rng = np.random.default_rng(seed)
    # Init: quantile means, common variance, uniform transitions
    qs = np.quantile(y, np.linspace(0.15, 0.85, K))
    mu = qs + rng.normal(0, 0.01 * y.std(), K)
    sigma = np.full(K, y.std())
    pi = np.full(K, 1 / K)
    P = np.full((K, K), 0.1 / (K - 1) if K > 1 else 1.0); np.fill_diagonal(P, 0.9)
    ll_prev = -np.inf
    for it in range(max_iter):
        log_emis = np.column_stack([stats.norm.logpdf(y, mu[k], sigma[k]) for k in range(K)])
        log_P = np.log(P + 1e-300); log_pi = np.log(pi + 1e-300)
        gamma, xi, ll = _forward_backward(log_emis, log_pi, log_P)
        # M-step
        pi = gamma[0] / gamma[0].sum()
        P = xi.sum(0) / gamma[:-1].sum(0)[:, None]
        for k in range(K):
            w = gamma[:, k]; W = w.sum()
            mu[k] = (w * y).sum() / W
            sigma[k] = math.sqrt(max(1e-8, (w * (y - mu[k]) ** 2).sum() / W))
        if abs(ll - ll_prev) < tol: break
        ll_prev = ll
    # Sort regimes by mu for consistency
    order = np.argsort(mu)
    mu = mu[order]; sigma = sigma[order]
    P = P[np.ix_(order, order)]; pi = pi[order]
    gamma = gamma[:, order]
    return {"mu": mu, "sigma": sigma, "transition_matrix": P, "initial_dist": pi,
            "smoothed_regime_prob": gamma,
            "log_likelihood": float(ll), "iterations": int(it + 1),
            "K": int(K), "T": int(T),
            "method": "Markov-switching Gaussian (Hamilton 1989) fit by EM"}


if __name__ == "__main__":
    rng = np.random.default_rng(2)
    T = 500
    # Simulate a 2-regime series: regime 0 = low vol N(0, 0.5), regime 1 = high vol N(0, 2.5)
    S = np.zeros(T, dtype=int)
    P_true = np.array([[0.95, 0.05], [0.10, 0.90]])
    for t in range(1, T):
        S[t] = rng.choice(2, p=P_true[S[t - 1]])
    y = np.where(S == 0, rng.normal(0, 0.5, T), rng.normal(0, 2.5, T))
    # Give the model a mean-switching flavor too
    y = y + np.where(S == 0, 0.0, 1.0)

    print("=== Markov-switching Gaussian, K=2 ===")
    r = markov_switching_normal(y, K=2, seed=0)
    print(f"  regime means:     {r['mu'].round(3)}")
    print(f"  regime std devs:  {r['sigma'].round(3)}")
    print(f"  transition matrix:\n{np.round(r['transition_matrix'], 3)}")
    print(f"  initial dist:     {r['initial_dist'].round(3)}")
    print(f"  log-lik: {r['log_likelihood']:.3f}, iters: {r['iterations']}")

    # Classification accuracy
    pred_S = r["smoothed_regime_prob"].argmax(1)
    acc = max(float((pred_S == S).mean()), float((pred_S == 1 - S).mean()))
    print(f"  smoothed classification accuracy: {acc:.3f}")

    print("\n--- library cross-check (statsmodels MarkovRegression) ---")
    try:
        from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
        m = MarkovRegression(y, k_regimes=2, switching_variance=True).fit(disp=False)
        print(f"  statsmodels regime means: {sorted(m.params[:2].round(3))}")
        print(f"  statsmodels regime std devs: {sorted(np.sqrt(m.params[-2:]).round(3))}")
    except Exception as ex:
        print(f"  (statsmodels not available: {ex})")
