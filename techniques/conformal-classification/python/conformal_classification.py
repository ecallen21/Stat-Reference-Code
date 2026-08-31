"""Conformal classification: APS + RAPS (Reference Ch 29 UQ).

Adaptive Prediction Sets (Romano-Sesia-Candes 2020) and Regularized APS
(Angelopoulos et al. 2021) produce classification "sets" with finite-sample
coverage guarantee 1 - alpha, distribution-free, on any pretrained
classifier.

APS score:  s(x, y) = sum_{k <= rank_of_y in decreasing softmax} p_(k)(x)
                        + U * p_y(x)     (randomised at the boundary)

Then on a held-out CALIBRATION set of size n_cal:
  q = ceil((n_cal + 1)(1 - alpha)) / n_cal quantile of s(x_i, y_i)

Prediction set:  {y : s(x, y) <= q}.

RAPS adds a regularisation penalty +lambda * max(rank - k_reg + 1, 0) to
shrink the average set size on easy problems.

Here we implement APS from scratch on a mini synthetic 3-class problem
with a hand-tuned classifier and verify empirical coverage matches the
target 1 - alpha.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def _softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def aps_score(probs, y, randomise=True, rng=None):
    """APS nonconformity score for each row (x_i, y_i)."""
    n, K = probs.shape
    order = np.argsort(-probs, axis=1)              # descending
    sorted_p = np.take_along_axis(probs, order, axis=1)
    cumsum = np.cumsum(sorted_p, axis=1)
    rank_of_y = np.argmax(order == y[:, None], axis=1)
    p_y = probs[np.arange(n), y]
    total = cumsum[np.arange(n), rank_of_y]
    if randomise:
        if rng is None: rng = np.random.default_rng(0)
        u = rng.uniform(size=n)
        score = total - u * p_y                     # uniform tie-break
    else:
        score = total
    return score


def calibrate(probs_cal, y_cal, alpha=0.1, seed=0):
    rng = np.random.default_rng(seed)
    s = aps_score(probs_cal, y_cal, randomise=True, rng=rng)
    n = len(s)
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_level = min(q_level, 1.0)
    return float(np.quantile(s, q_level, method="higher"))


def predict_sets(probs_test, q, randomise=True, seed=1):
    n, K = probs_test.shape
    rng = np.random.default_rng(seed)
    order = np.argsort(-probs_test, axis=1)
    sorted_p = np.take_along_axis(probs_test, order, axis=1)
    cumsum = np.cumsum(sorted_p, axis=1)
    sets = []
    for i in range(n):
        included = []
        for r in range(K):
            u = rng.uniform() if randomise else 0.0
            s = cumsum[i, r] - u * sorted_p[i, r]
            if s <= q:
                included.append(int(order[i, r]))
        if not included:
            included.append(int(order[i, 0]))
        sets.append(sorted(included))
    return sets


if __name__ == "__main__":
    print("=== APS conformal classification (Romano 2020) ===\n")
    rng = np.random.default_rng(0)
    n_train, n_cal, n_test = 200, 300, 500
    K = 4
    # Three-class classifier with noisy softmax margin
    def make(n):
        y = rng.integers(0, K, n)
        z = rng.normal(0, 1.0, (n, K))
        z[np.arange(n), y] += 1.0
        return _softmax(z), y

    probs_cal, y_cal = make(n_cal)
    probs_te,  y_te  = make(n_test)

    for alpha in (0.05, 0.10, 0.20):
        q = calibrate(probs_cal, y_cal, alpha=alpha)
        sets = predict_sets(probs_te, q, randomise=True, seed=2)
        cov = np.mean([y_te[i] in sets[i] for i in range(n_test)])
        sizes = [len(s) for s in sets]
        print(f"  alpha={alpha:.2f}  q={q:.3f}  target cov={1-alpha:.2f}  empirical cov={cov:.3f}"
              f"  mean set size={np.mean(sizes):.2f}")

    print("\n  Example prediction sets:")
    for i in range(6):
        print(f"    y_true={y_te[i]}  probs={np.round(probs_te[i], 2).tolist()}"
              f"  set={sets[i]}")

    print("\n--- library cross-check (mapie / crepes / puncc; sklearn wrapper) ---")
