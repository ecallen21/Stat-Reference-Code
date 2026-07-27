"""Bradley-Terry model for pairwise-comparison rankings (Reference §8.8).

Given items 1..K and observed wins in pairwise matchups, the Bradley-Terry model
assigns each item a positive "ability" parameter pi_i so that

    P(i beats j)  =  pi_i / (pi_i + pi_j)

Equivalent logistic-regression form with beta_i = log(pi_i):

    logit P(i beats j)  =  beta_i - beta_j

with the identifiability constraint sum(beta) = 0 (or equivalently pi_K = 1).

Fitting via the MM (minorize-maximize) algorithm of Ford (1957) / Hunter (2004):
    pi_i <- W_i / sum_{j != i} n_ij / (pi_i + pi_j)
where W_i = wins by i, n_ij = total games between i and j.

Guaranteed monotone convergence; no derivative needed. Alternate normalisation
each iteration to prevent runoff.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy import stats    # distributions, hypothesis tests, PPFs (norm, t, chi2, ttest_ind, ...)


def fit_bradley_terry(win_matrix, item_labels=None,
                       max_iter: int = 1000, tol: float = 1e-10) -> dict:
    """MM algorithm for the Bradley-Terry model.

    Parameters
    ----------
    win_matrix : K x K integer array W where W[i, j] = number of times i beat j.
    item_labels : optional K names for the items.
    max_iter, tol : stopping criteria on the relative change in pi.
    """
    W = np.asarray(win_matrix, dtype=float)
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError("win_matrix must be square (K x K)")
    K = W.shape[0]
    if item_labels is None:
        item_labels = [f"Item{i}" for i in range(K)]
    Wi = W.sum(axis=1)                             # total wins by i
    Nij = W + W.T                                   # total games between i and j
    # start uniform
    pi = np.ones(K)
    for it in range(max_iter):
        # For each item i, denom_i = sum_{j != i} n_ij / (pi_i + pi_j)
        # Vectorize:
        pi_new = np.empty(K)
        for i in range(K):
            denom = 0.0
            for j in range(K):
                if i == j: continue
                if Nij[i, j] == 0: continue
                denom += Nij[i, j] / (pi[i] + pi[j])
            if Wi[i] == 0 or denom == 0:
                pi_new[i] = 1e-12
            else:
                pi_new[i] = Wi[i] / denom
        # Normalize so the geometric mean is 1 -- prevents runoff, gives a
        # symmetric interpretation for exp of beta = log pi.
        gmean = np.exp(np.mean(np.log(np.clip(pi_new, 1e-300, None))))
        pi_new = pi_new / gmean
        if np.max(np.abs(pi_new - pi) / np.clip(pi, 1e-12, None)) < tol:
            pi = pi_new; break
        pi = pi_new
    # Wald SEs via Fisher information at the MLE (see Hunter 2004 / Agresti)
    # The information for beta = log(pi) is J_ii = sum_{j != i} n_ij pi_i pi_j / (pi_i + pi_j)^2
    beta = np.log(pi)
    J = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            if i == j: continue
            if Nij[i, j] == 0: continue
            v = Nij[i, j] * pi[i] * pi[j] / (pi[i] + pi[j]) ** 2
            J[i, i] += v
            J[i, j] -= v
    # Project out the sum(beta) = 0 constraint by using the (K-1) x (K-1)
    # submatrix's pseudoinverse.
    try:
        # Add a small ridge to make invertible under the constraint
        cov_beta = np.linalg.pinv(J)
        se_beta = np.sqrt(np.clip(np.diag(cov_beta), 0.0, None))
    except np.linalg.LinAlgError:
        se_beta = np.full(K, np.nan)
    # Ranking
    order = np.argsort(-pi)
    return {"pi": pi.tolist(),
            "beta_log_pi": beta.tolist(),
            "SE_beta": se_beta.tolist(),
            "item_labels": list(item_labels),
            "ranking_descending": [item_labels[o] for o in order],
            "n_iter": it + 1,
            "converged": it < max_iter - 1,
            "method": "Bradley-Terry MLE via MM algorithm (Hunter 2004)"}


def predict_win_prob(fit, i, j):
    """P(item i beats item j) under the fitted BT model. i, j are indices or labels."""
    labels = fit["item_labels"]
    ii = i if isinstance(i, int) else labels.index(i)
    jj = j if isinstance(j, int) else labels.index(j)
    pi = fit["pi"]
    return pi[ii] / (pi[ii] + pi[jj])


def library_versions(win_matrix):
    """statsmodels doesn't ship BT out of the box; try the choix package if present."""
    try:
        import choix
        K = np.asarray(win_matrix).shape[0]
        data = []
        for i in range(K):
            for j in range(K):
                if i == j: continue
                for _ in range(int(win_matrix[i][j])):
                    data.append((i, j))     # i beat j
        params = choix.ilsr_pairwise(K, data, alpha=1e-6)
        return {"choix ilsr (log-scale, may differ by additive constant)":
                params.tolist()}
    except Exception as ex:
        return {"choix (optional)": f"not available: {ex}"}


if __name__ == "__main__":
    # 5 chess players, W[i,j] = wins of i over j
    labels = ["Alice", "Bob", "Carol", "Dan", "Eve"]
    W = np.array([
        [ 0,  4,  6,  8, 10],   # Alice
        [ 2,  0,  3,  5,  7],   # Bob
        [ 1,  3,  0,  4,  6],   # Carol
        [ 1,  2,  3,  0,  5],   # Dan
        [ 0,  1,  2,  2,  0],   # Eve
    ], dtype=float)

    fit = fit_bradley_terry(W, labels)
    print("=== Bradley-Terry fit ===")
    print(f"  converged in {fit['n_iter']} iterations")
    print(f"  ranking: {fit['ranking_descending']}")
    print(f"  pi (ability, geometric-mean = 1):")
    for lbl, p in zip(fit["item_labels"], fit["pi"]):
        print(f"    {lbl:7s}: {p:.4f}")
    print(f"  beta = log(pi):")
    for lbl, b, s in zip(fit["item_labels"], fit["beta_log_pi"], fit["SE_beta"]):
        print(f"    {lbl:7s}: beta={b:+.4f}  SE={s:.4f}")

    print("\n=== Predicted matchup probabilities ===")
    for a, b in [("Alice", "Bob"), ("Alice", "Eve"), ("Dan", "Eve")]:
        p = predict_win_prob(fit, a, b)
        print(f"  P({a} beats {b}) = {p:.3f}")

    print("\n--- library ---")
    for k, v in library_versions(W).items():
        print(f"  {k}: {v}")
