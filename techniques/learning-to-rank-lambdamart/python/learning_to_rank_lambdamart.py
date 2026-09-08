"""LambdaMART learning-to-rank (Reference Sec 47.129).

Burges 2010 'From RankNet to LambdaRank to LambdaMART: An
overview', Microsoft Tech Report. Combines:

  * pairwise cross-entropy loss (RankNet)
  * scale each pairwise gradient by |Delta NDCG_{i,j}| (LambdaRank)
  * fit gradient-boosted trees on the aggregated 'lambda' gradients
    (LambdaMART).

The gradient at document i in query q:

    lambda_i = sum_{j: rel(i) != rel(j)}  sign(rel(i) - rel(j)) *
                sigmoid(-|score_i - score_j|) * |Delta NDCG_{ij}|.

Illustrated with a small toy: three queries, per-query relevance
labels, ranker = per-feature linear model tuned via lambda gradient.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def dcg_at_k(rels, k=None):
    if k is None: k = len(rels)
    r = np.asarray(rels)[:k]
    gains = (2.0 ** r - 1)
    disc = 1.0 / np.log2(np.arange(2, len(r) + 2))
    return float((gains * disc).sum())


def ndcg_at_k(rels_pred, rels_ideal, k=None):
    return dcg_at_k(rels_pred, k) / max(dcg_at_k(rels_ideal, k), 1e-12)


def lambda_gradients(scores, rels):
    n = len(scores)
    sigma = 1.0
    ideal = np.sort(rels)[::-1]
    idcg = max(dcg_at_k(ideal), 1e-12)
    order = np.argsort(-scores)
    rank = np.empty(n, dtype=int); rank[order] = np.arange(n)
    disc = 1.0 / np.log2(np.arange(2, n + 2))
    lam = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if rels[i] <= rels[j]: continue
            # Swap gain in DCG
            gain_i = 2.0 ** rels[i] - 1; gain_j = 2.0 ** rels[j] - 1
            delta_ndcg = abs(disc[rank[i]] - disc[rank[j]]) * (gain_i - gain_j) / idcg
            rho = 1.0 / (1.0 + np.exp(sigma * (scores[i] - scores[j])))
            lam[i] -= sigma * rho * delta_ndcg
            lam[j] += sigma * rho * delta_ndcg
    return lam


if __name__ == "__main__":
    print("=== LambdaMART learning-to-rank (Burges 2010) ===\n")
    rng = np.random.default_rng(0)
    Q, D, d = 8, 10, 5      # queries, docs per query, features
    Xs = [rng.normal(size=(D, d)) for _ in range(Q)]
    beta_true = rng.normal(size=d)
    Ys = []
    for X in Xs:
        z = X @ beta_true
        r = np.digitize(z, np.quantile(z, [0.4, 0.7]))
        Ys.append(r)

    # LambdaMART lite: gradient boosting on lambda gradients using linear regressors
    from sklearn.linear_model import LinearRegression
    boosters = []
    lr = 0.3
    def predict(X):
        s = np.zeros(len(X))
        for m in boosters:
            s += lr * m.predict(X)
        return s
    for _ in range(50):
        Xs_stack, lam_stack = [], []
        for X, r in zip(Xs, Ys):
            scores = predict(X)
            lam = lambda_gradients(scores, r)
            Xs_stack.append(X); lam_stack.append(-lam)      # neg-grad for regression
        m = LinearRegression().fit(np.vstack(Xs_stack), np.concatenate(lam_stack))
        boosters.append(m)

    ndcgs = []
    for X, r in zip(Xs, Ys):
        s = predict(X)
        order = np.argsort(-s)
        ideal = np.sort(r)[::-1]
        ndcgs.append(ndcg_at_k(r[order], ideal))
    print(f"  Trained on {Q} queries, {D} docs each, {d} features")
    print(f"  Mean NDCG (LambdaMART lite)       = {np.mean(ndcgs):.3f}")

    # Baseline: random ordering
    rand_ndcgs = []
    for X, r in zip(Xs, Ys):
        order = rng.permutation(D)
        ideal = np.sort(r)[::-1]
        rand_ndcgs.append(ndcg_at_k(r[order], ideal))
    print(f"  Random ordering baseline         = {np.mean(rand_ndcgs):.3f}")

    print("\n--- library cross-check (lightgbm.LGBMRanker Python; xgboost.XGBRanker; ranger R) ---")
