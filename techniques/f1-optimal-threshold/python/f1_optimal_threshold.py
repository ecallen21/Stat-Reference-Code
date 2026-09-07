"""F1-optimal decision threshold (Reference Sec 26.9).

For a binary classifier with probabilistic output p in [0, 1], the
decision rule is I{p > t}. Rather than using t = 0.5 (which is
Bayes-optimal only for balanced classes), pick the threshold t that
MAXIMISES the F1 SCORE (or F_beta) on a validation set:

    F_beta = (1 + beta^2) * P * R / (beta^2 * P + R)

Sweep t in [0, 1], compute (precision, recall, F_beta) and return the
argmax. Especially important for CLASS-IMBALANCED problems where
t = 0.5 gives near-zero recall for the minority class.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def f_beta_score(y_true, y_pred, beta=1.0):
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    if tp == 0:
        return 0.0
    P = tp / (tp + fp); R = tp / (tp + fn)
    return (1 + beta ** 2) * P * R / (beta ** 2 * P + R)


def find_optimal_threshold(y_true, prob, beta=1.0, n_grid=101):
    grid = np.linspace(0.01, 0.99, n_grid)
    scores = np.array([f_beta_score(y_true, (prob >= t).astype(int), beta=beta) for t in grid])
    idx = int(np.argmax(scores))
    return {"t_opt": float(grid[idx]), "F_opt": float(scores[idx]),
            "grid": grid, "F_grid": scores}


if __name__ == "__main__":
    print("=== F1 (F-beta) optimal decision threshold ===\n")
    rng = np.random.default_rng(0)
    n = 3000
    #  Class imbalance: y=1 in 5% of cases
    y = rng.binomial(1, 0.05, size=n)
    #  Score: N(y * 2, 1) so class 1 tends higher
    score = rng.normal(loc=2 * y, scale=1)
    prob = 1 / (1 + np.exp(-(score - 1.0)))    # calibrated-ish

    print(f"  Positive class prevalence = {y.mean():.2%}")

    for beta in [0.5, 1.0, 2.0]:
        r = find_optimal_threshold(y, prob, beta=beta)
        print(f"\n  beta = {beta}   (F_beta favours "
              f"{'precision' if beta < 1 else 'recall' if beta > 1 else 'balance'}):")
        print(f"    t* = {r['t_opt']:.3f}   F_beta(t*) = {r['F_opt']:.3f}")
        print(f"    F_beta at t = 0.5        = "
              f"{f_beta_score(y, (prob >= 0.5).astype(int), beta=beta):.3f}   "
              f"(naive default)")

    print("\n--- library cross-check (yardstick / probably R; sklearn.metrics.fbeta_score Python) ---")
