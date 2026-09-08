"""Plackett-Luce ranking model (Reference Sec 47.53).

Plackett 1975 'The analysis of permutations', J R Stat Soc C 24(2);
Luce 1959 'Individual Choice Behavior'. For rankings over K items,

    P(pi_1, ..., pi_K)  =  prod_{k=1}^{K}  worth[pi_k] / sum_{j >= k} worth[pi_j]

with 'worth' parameters w_i > 0. Reduces to Bradley-Terry when
K = 2. Fit by MM iteration (Hunter 2004):

    w_i^{(t+1)} = W_i / sum over rankings r, positions k <= last(i)
                        of  1 / sum_{j in denom_r,k} w_j^{(t)}

where W_i = # rankings in which item i beat at least one other.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def fit_plackett_luce(rankings, n_items, max_iter=500, tol=1e-6):
    """MM algorithm (Hunter 2004) for Plackett-Luce worths.

    `rankings` is a list of orderings (item indices, best first).
    """
    w = np.ones(n_items)
    for it in range(max_iter):
        num = np.zeros(n_items)
        for r in rankings:
            for i in r[:-1]:
                num[i] += 1
        denom = np.zeros(n_items)
        for r in rankings:
            remaining = list(r)
            for pos in range(len(r) - 1):
                s = sum(w[j] for j in remaining)
                if s > 0:
                    for j in remaining:
                        denom[j] += 1.0 / s
                remaining.pop(0)
        w_new = np.where(denom > 0, num / denom, w)
        w_new = w_new * n_items / w_new.sum()    # normalise
        if np.max(np.abs(w_new - w)) < tol:
            w = w_new
            break
        w = w_new
    return {"worth": w, "iters": it + 1}


if __name__ == "__main__":
    print("=== Plackett-Luce ranking model (Plackett 1975; Luce 1959) ===\n")
    rng = np.random.default_rng(0)
    n_items, n_rank = 5, 800
    true_w = np.array([4.0, 3.0, 2.0, 1.0, 0.5])
    rankings = []
    for _ in range(n_rank):
        # Simulate one ranking via sequential Luce
        available = list(range(n_items))
        weights = true_w.copy()
        order = []
        for _ in range(n_items):
            p = weights[available] / weights[available].sum()
            pick = int(rng.choice(available, p=p))
            order.append(pick); available.remove(pick)
        rankings.append(order)

    fit = fit_plackett_luce(rankings, n_items)
    w_hat = fit["worth"]
    # Normalise both to sum n_items for comparability
    tw = true_w * n_items / true_w.sum()
    print(f"  Converged in {fit['iters']} iterations.")
    print(f"  True worth  (normalised): {np.round(tw, 3)}")
    print(f"  Estim worth (normalised): {np.round(w_hat, 3)}")
    print(f"  Rank order recovered:     {list(np.argsort(w_hat)[::-1])}")
    print(f"  Rank order truth:         {list(np.argsort(true_w)[::-1])}")

    print("\n--- library cross-check (PlackettLuce R; choix Python) ---")
