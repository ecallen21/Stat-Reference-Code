"""Matthews Correlation Coefficient (Reference Sec 47.253).

Matthews 1975 'Comparison of the Predicted and Observed Secondary
Structure of T4 Phage Lysozyme', BBA. A BALANCED classification
metric that stays informative under class imbalance:

    MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))

Range [-1, +1]. +1 = perfect, 0 = random, -1 = perfect inverse.
Unlike accuracy or F1, MCC treats classes symmetrically and
penalises trivial majority-only predictions.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def mcc(y_true, y_pred):
    """Matthews correlation coefficient (binary)."""
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denom == 0: return 0.0
    return (tp * tn - fp * fn) / denom


if __name__ == "__main__":
    print("=== Matthews Correlation Coefficient (Matthews 1975) ===\n")
    rng = np.random.default_rng(0)

    # Highly imbalanced (99/1); trivial majority prediction inflates accuracy
    n = 10000
    y = np.zeros(n, dtype=int); y[:100] = 1                     # 1% positive
    rng.shuffle(y)

    print(f"  n = {n:,}, positive fraction = {y.mean() * 100:.1f}%\n")
    scenarios = [
        ("all-neg (trivial)",  np.zeros(n, dtype=int)),
        ("all-pos",             np.ones(n, dtype=int)),
        ("random",              rng.integers(0, 2, size=n)),
        ("weak positive",       (y | (rng.random(n) < 0.02)).astype(int)),  # catches all + noise
        ("perfect",             y.copy()),
    ]
    print(f"  {'model':<22}  {'accuracy':>10}  {'F1':>8}  {'MCC':>8}")
    from sklearn.metrics import f1_score, accuracy_score, matthews_corrcoef
    for name, yhat in scenarios:
        acc = accuracy_score(y, yhat)
        f1  = f1_score(y, yhat, zero_division=0)
        m   = mcc(y, yhat)
        print(f"  {name:<22}  {acc:>10.4f}  {f1:>8.3f}  {m:>8.3f}")

    # sanity vs sklearn
    print(f"\n  Our MCC(perfect) = {mcc(y, y):.4f}   sklearn = {matthews_corrcoef(y, y):.4f}")
    yhat = rng.integers(0, 2, size=n)
    print(f"  Our MCC(random)  = {mcc(y, yhat):.4f}   sklearn = {matthews_corrcoef(y, yhat):.4f}")

    print("\n  MCC exposes trivial classifiers: 'all-neg' gets acc=0.99 but MCC=0.")
    print("  Now recommended over F1 for imbalanced binary tasks (Chicco & Jurman 2020).")

    print("\n--- library cross-check (sklearn.metrics.matthews_corrcoef) ---")
