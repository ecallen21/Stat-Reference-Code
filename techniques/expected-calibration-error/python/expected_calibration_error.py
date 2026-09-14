"""Expected Calibration Error - ECE (Reference Sec 47.283).

Naeini, Cooper & Hauskrecht 2015 AAAI; Guo et al 2017 ICML.
Weighted average absolute gap between confidence and accuracy
across confidence bins:

    ECE = sum_m (|B_m| / n) * |acc(B_m) - conf(B_m)|
    MCE = max_m |acc(B_m) - conf(B_m)|      (worst-bin gap)
    ACE = adaptive-binning ECE (equal-mass bins)

Standard metric for calibration. Complements NLL / Brier and
reliability diagrams.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def compute_ece(confs, correct, n_bins=15, adaptive=False):
    """ECE and MCE. If adaptive=True use equal-mass bins."""
    if adaptive:
        edges = np.quantile(confs, np.linspace(0, 1, n_bins + 1))
    else:
        edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0; mce = 0.0; rows = []
    for i in range(n_bins):
        if i == n_bins - 1:
            m = (confs >= edges[i]) & (confs <= edges[i + 1])
        else:
            m = (confs >= edges[i]) & (confs < edges[i + 1])
        if m.sum() == 0: continue
        acc = correct[m].mean(); conf = confs[m].mean()
        gap = abs(acc - conf)
        ece += (m.sum() / len(confs)) * gap
        mce = max(mce, gap)
        rows.append((i, m.sum(), conf, acc, gap))
    return float(ece), float(mce), rows


if __name__ == "__main__":
    print("=== Expected Calibration Error (Naeini 2015; Guo 2017) ===\n")
    rng = np.random.default_rng(0)

    # Simulate overconfident model: reported confidences are too high
    n = 4000
    true_prob = rng.beta(2, 5, n)                                 # true P(correct | x)
    confs_bad = np.clip(true_prob * 1.5, 0.01, 0.99)              # overconfident
    correct = (rng.uniform(size=n) < true_prob).astype(float)

    ece_bad, mce_bad, rows = compute_ece(confs_bad, correct, n_bins=10)
    print(f"  Uncalibrated (overconfident) model, n = {n}:")
    print(f"    ECE = {ece_bad:.4f}   MCE = {mce_bad:.4f}\n")
    print(f"    {'bin':>3}  {'n':>5}  {'conf':>6}  {'acc':>6}  {'gap':>6}")
    for i, cnt, conf, acc, gap in rows:
        print(f"    {i:>3}  {cnt:>5}  {conf:>6.3f}  {acc:>6.3f}  {gap:>6.3f}")

    # Calibrated (confidence = true probability)
    ece_ok, mce_ok, _ = compute_ece(true_prob, correct, n_bins=10)
    print(f"\n  Well-calibrated (confs = true prob):")
    print(f"    ECE = {ece_ok:.4f}   MCE = {mce_ok:.4f}\n")

    # Adaptive ECE (equal-mass bins)
    ece_a, mce_a, _ = compute_ece(confs_bad, correct, n_bins=10, adaptive=True)
    print(f"  Adaptive ECE (uncalibrated): ECE = {ece_a:.4f}, MCE = {mce_a:.4f}")
    print(f"  Adaptive binning avoids empty / near-empty bins at the extremes;")
    print(f"  equal-width ECE can be misleading when confidences are concentrated.")

    print("\n--- library cross-check (netcal.metrics.ECE / MCE; torchmetrics.CalibrationError) ---")
