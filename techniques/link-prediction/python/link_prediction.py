"""Link prediction by similarity scores (Reference §24.7).

Score every non-existing pair (i, j) and rank candidate edges.  Five classical
neighbourhood-based similarities:

  * Common neighbours:      |N(i) ∩ N(j)|
  * Jaccard:                |N(i) ∩ N(j)| / |N(i) ∪ N(j)|
  * Adamic-Adar:            sum_{z in N(i) ∩ N(j)} 1 / log(deg(z))
  * Resource allocation:    sum_{z in N(i) ∩ N(j)} 1 / deg(z)
  * Preferential attach:    deg(i) * deg(j)

Evaluation: split edges into train / test; score all node-pairs not in the
train graph; compute ROC-AUC of the score against the "is this a held-out
edge" indicator.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

import numpy as np    # numerical arrays + linear algebra


def _neighbours(A):
    return [set(np.where(row)[0]) for row in A]


def score_pair(A, i, j, method: str = "adamic_adar") -> float:
    N = _neighbours(A)
    d = A.sum(axis=1)
    common = N[i] & N[j]
    if method == "common":
        return float(len(common))
    if method == "jaccard":
        u = N[i] | N[j]; return len(common) / max(len(u), 1)
    if method == "adamic_adar":
        return float(sum(1.0 / np.log(d[z]) for z in common if d[z] > 1))
    if method == "resource_allocation":
        return float(sum(1.0 / d[z] for z in common if d[z] > 0))
    if method == "preferential":
        return float(d[i] * d[j])
    raise ValueError(f"unknown method '{method}'")


def _all_pairs_scores(A, method):
    A = np.asarray(A); n = A.shape[0]
    d = A.sum(axis=1)
    N = _neighbours(A)
    S = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if method == "common":
                s = len(N[i] & N[j])
            elif method == "jaccard":
                u = N[i] | N[j]
                s = len(N[i] & N[j]) / max(len(u), 1)
            elif method == "adamic_adar":
                s = sum(1.0 / np.log(d[z]) for z in N[i] & N[j] if d[z] > 1)
            elif method == "resource_allocation":
                s = sum(1.0 / d[z] for z in N[i] & N[j] if d[z] > 0)
            elif method == "preferential":
                s = d[i] * d[j]
            else:
                raise ValueError(method)
            S[i, j] = S[j, i] = s
    return S


def roc_auc(labels, scores) -> float:
    """Rank-based AUC via the Mann-Whitney formulation."""
    labels = np.asarray(labels); scores = np.asarray(scores)
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # handle ties: average ranks for equal scores
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts)); np.add.at(sums, inv, ranks)
    avg = sums / counts
    ranks_tied = avg[inv]
    pos = ranks_tied[labels == 1]
    n_pos = (labels == 1).sum(); n_neg = (labels == 0).sum()
    U = pos.sum() - n_pos * (n_pos + 1) / 2.0
    return float(U / (n_pos * n_neg)) if n_pos and n_neg else float("nan")


def link_prediction_eval(A_full, test_frac: float = 0.2,
                          seed: int = 0) -> dict:
    """Hide test_frac of edges, score non-train pairs, report AUC per method."""
    rng = np.random.default_rng(seed)
    A = A_full.copy(); n = A.shape[0]
    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if A[i, j]]
    rng.shuffle(edges)
    n_test = int(len(edges) * test_frac)
    test = edges[:n_test]
    A_train = A.copy()
    for i, j in test:
        A_train[i, j] = A_train[j, i] = 0

    # candidate pairs: non-edges in train graph
    pairs = []; labels = []
    train_edge_set = {(i, j) for i in range(n) for j in range(i + 1, n) if A_train[i, j]}
    test_set = {(i, j) for i, j in test}
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) in train_edge_set:
                continue
            pairs.append((i, j))
            labels.append(1 if (i, j) in test_set else 0)
    labels = np.asarray(labels, dtype=int)

    out = {}
    for m in ("common", "jaccard", "adamic_adar", "resource_allocation", "preferential"):
        S = _all_pairs_scores(A_train, m)
        s = np.array([S[i, j] for i, j in pairs])
        out[m] = roc_auc(labels, s)
    return {"auc": out, "n_test_edges": int(n_test),
            "n_candidate_pairs": len(pairs),
            "method": "held-out link prediction (5 similarity scores)"}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # planted-partition graph: within-p 0.4, between-p 0.02 (clear community signal)
    sizes = [20, 20, 20]; n = sum(sizes)
    z = np.repeat(range(len(sizes)), sizes)
    A = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            p = 0.4 if z[i] == z[j] else 0.02
            A[i, j] = A[j, i] = int(rng.uniform() < p)

    res = link_prediction_eval(A, test_frac=0.2, seed=1)
    print(f"=== Link prediction on planted-partition (n=60, K=3) ===")
    print(f"  hidden edges = {res['n_test_edges']},  candidate pairs = {res['n_candidate_pairs']}")
    print(f"  {'method':>22}  {'AUC':>6}")
    for m, a in res["auc"].items():
        print(f"  {m:>22}  {a:>6.3f}")

    print("\n--- library cross-check (networkx.link_prediction) ---")
    try:
        import networkx as nx
        # take Adamic-Adar as spot-check
        G = nx.from_numpy_array(A)
        aa = dict(((u, v), p) for u, v, p in nx.adamic_adar_index(G))
        print(f"  nx.adamic_adar_index computed for {len(aa)} candidate pairs")
    except ImportError:
        print("  (networkx not installed)")
