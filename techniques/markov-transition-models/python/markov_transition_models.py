"""Markov Transition Models for Longitudinal Categorical Data (Reference §12.9).

For a discrete-state variable observed repeatedly per subject (healthy/sick,
low/med/high risk, employed/unemployed), a first-order Markov model assumes:

    P(Y_{t+1} = j | Y_t = i, history)  =  P(Y_{t+1} = j | Y_t = i)  =  P_ij

Estimator: MLE of the K x K transition-probability matrix P is simply the
proportion of transitions from i to j observed in the data:

    P_ij_hat  =  n_ij / n_i.       where n_ij = # times i -> j transitions occurred,
                                          n_i. = # times state i was followed by anything.

Related quantities:
    - Stationary distribution : long-run proportion of time in each state,
      pi solving pi = pi P (leading left eigenvector of P for eigenvalue 1).
    - Order test (§12.9)      : LR test H0: first-order vs H1: second-order
      based on the log-likelihoods of two nested transition matrices.

Extensions (not implemented here):
    - Non-homogeneous transitions (P depends on t) -- fit separate P per period.
    - Transitions depending on covariates -- fit multinomial GLM per row of P.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy import stats    # distributions, hypothesis tests, PPFs (norm, t, chi2, ttest_ind, ...)


def fit_transition_matrix(sequences, states=None) -> dict:
    """MLE of a first-order K x K transition-probability matrix.

    Parameters
    ----------
    sequences : list of per-subject sequences (each a list/array of state labels).
    states    : optional ordered list of state labels; inferred if None.
    """
    if states is None:
        states = sorted({s for seq in sequences for s in seq})
    K = len(states); idx = {s: i for i, s in enumerate(states)}
    N = np.zeros((K, K), dtype=int)                    # transition counts
    for seq in sequences:
        for a, b in zip(seq[:-1], seq[1:]):
            N[idx[a], idx[b]] += 1
    row_totals = N.sum(axis=1, keepdims=True)
    P = np.where(row_totals > 0, N / np.clip(row_totals, 1, None), 0.0)
    # Stationary distribution: left eigenvector for eigenvalue 1
    w, v = np.linalg.eig(P.T)
    # Pick eigenvalue closest to 1
    stationary_idx = int(np.argmin(np.abs(w - 1)))
    stationary = np.real(v[:, stationary_idx])
    stationary = stationary / stationary.sum()
    return {"states": list(states),
            "counts": N.tolist(),
            "P_transition": P.tolist(),
            "row_totals": row_totals.flatten().tolist(),
            "stationary_distribution": stationary.tolist(),
            "n_subjects": len(sequences),
            "n_transitions": int(N.sum()),
            "method": "MLE first-order transition matrix + stationary distribution"}


def test_first_order_vs_zero_order(sequences, states=None) -> dict:
    """LR test H0: transitions are independent of previous state (zero-order,
    i.e. marginal P(Y_t = j)) vs H1: first-order Markov.

    LR = 2 * (ll_first_order - ll_zero_order)  ~  chi^2 with (K - 1)^2 df
    """
    if states is None:
        states = sorted({s for seq in sequences for s in seq})
    K = len(states); idx = {s: i for i, s in enumerate(states)}
    # First-order: log-lik under fitted P
    N = np.zeros((K, K), dtype=int)
    for seq in sequences:
        for a, b in zip(seq[:-1], seq[1:]):
            N[idx[a], idx[b]] += 1
    row_totals = N.sum(axis=1, keepdims=True)
    P = np.where(row_totals > 0, N / np.clip(row_totals, 1, None), 0.0)
    ll_first = 0.0
    for i in range(K):
        for j in range(K):
            if N[i, j] > 0:
                ll_first += N[i, j] * math.log(max(P[i, j], 1e-12))
    # Zero-order: marginal probabilities over destinations
    col_totals = N.sum(axis=0)
    p_marg = col_totals / max(col_totals.sum(), 1)
    ll_zero = 0.0
    for i in range(K):
        for j in range(K):
            if N[i, j] > 0:
                ll_zero += N[i, j] * math.log(max(p_marg[j], 1e-12))
    lr = 2 * (ll_first - ll_zero)
    df = (K - 1) ** 2
    p = float(stats.chi2.sf(lr, df))
    return {"log_lik_first_order": ll_first,
            "log_lik_zero_order": ll_zero,
            "LR": lr, "df": df, "p_value": p,
            "interpretation": ("Small p => first-order Markov structure exists "
                                "(transitions depend on the previous state)"),
            "method": "LR test: first-order Markov vs. independence"}


if __name__ == "__main__":
    rng = np.random.default_rng(37)
    states = ["healthy", "at_risk", "diagnosed"]
    true_P = np.array([
        [0.85, 0.12, 0.03],       # healthy tends to stay healthy
        [0.20, 0.60, 0.20],       # at-risk in the middle
        [0.05, 0.10, 0.85],       # diagnosed tends to stay diagnosed
    ])
    # Simulate 300 subjects with 6 timepoints each
    def simulate_one(K=6):
        s = "healthy"; out = [s]
        for _ in range(K - 1):
            probs = true_P[states.index(s)]
            s = rng.choice(states, p=probs); out.append(s)
        return out
    sequences = [simulate_one() for _ in range(300)]

    print("=== Transition matrix MLE (true P shown for comparison) ===")
    fit = fit_transition_matrix(sequences, states)
    print(f"  states: {fit['states']}")
    print(f"  estimated P:")
    for i, row in enumerate(fit["P_transition"]):
        print(f"    from {states[i]:10s}: {[f'{v:.3f}' for v in row]}")
    print(f"  stationary distribution: {[f'{v:.3f}' for v in fit['stationary_distribution']]}")
    print(f"  n_transitions observed: {fit['n_transitions']}")

    print("\n=== LR test: first-order vs zero-order ===")
    lr = test_first_order_vs_zero_order(sequences, states)
    print(f"  LR = {lr['LR']:.2f}, df = {lr['df']}, p = {lr['p_value']:.4g}")
    print(f"  {lr['interpretation']}")
