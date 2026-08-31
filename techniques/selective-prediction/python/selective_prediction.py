"""Selective prediction / prediction with abstention (Reference Ch 29 UQ).

Chow (1957); El-Yaniv & Wiener (2010); Geifman & El-Yaniv (2017)
"Selective Classification for Deep Neural Networks."

Given a base classifier f and a confidence score g(x) in [0, 1]:

  Predict:  y_hat if g(x) >= tau, else ABSTAIN.

Two headline metrics vary with the threshold tau:

  Coverage  c(tau) = P( g(x) >= tau )                  (how often we act)
  Selective error  e(tau) = P( y != y_hat | g(x) >= tau )
  Selective accuracy = 1 - e(tau)

The RISK-COVERAGE curve traces (c(tau), e(tau)) as tau sweeps [0, 1];
the AURC (Area Under Risk-Coverage) is the standard summary metric.

Given a target *risk* r_max (say 1% error), pick the LARGEST tau with
  e(tau) <= r_max.  This maximises coverage under the risk cap.

Here we implement risk-coverage tracing + selective accuracy + AURC on a
softmax classifier's outputs. Confidence score = MSP (max softmax).
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def _softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def risk_coverage(y_true, y_pred, confidence):
    """Return (coverage, selective_error) sorted by decreasing confidence."""
    order = np.argsort(-confidence)
    correct = (y_pred == y_true).astype(float)[order]
    n = len(correct)
    cum_correct = np.cumsum(correct)
    ks = np.arange(1, n + 1)
    coverage = ks / n
    selective_err = 1.0 - cum_correct / ks
    return coverage, selective_err, confidence[order]


def aurc(coverage, selective_err):
    """Area Under Risk-Coverage curve (lower is better)."""
    return float(np.trapezoid(selective_err, coverage))


def coverage_at_risk(coverage, selective_err, r_max):
    """Largest coverage with selective error <= r_max."""
    mask = selective_err <= r_max
    return float(coverage[mask].max()) if mask.any() else 0.0


if __name__ == "__main__":
    print("=== Selective prediction / abstention (Geifman-El-Yaniv 2017) ===\n")
    rng = np.random.default_rng(0)
    n = 1000
    K = 4
    # Simulate a classifier: logits have varying margin
    y_true = rng.integers(0, K, n)
    margins = rng.exponential(1.5, n)                # >0
    z = rng.normal(0, 0.5, (n, K))
    z[np.arange(n), y_true] += margins
    # Add "hard" examples with tiny/wrong margin
    hard = rng.random(n) < 0.15
    z[hard] = rng.normal(0, 1.5, (hard.sum(), K))

    p = _softmax(z)
    y_pred = p.argmax(axis=1)
    confidence = p.max(axis=1)

    print(f"  full-coverage accuracy: {(y_pred == y_true).mean():.3f}")

    cov, err, conf_sorted = risk_coverage(y_true, y_pred, confidence)

    print("\n  Risk-Coverage table (selected thresholds):")
    print(f"    {'coverage':>9}  {'sel_err':>8}  {'sel_acc':>8}  {'threshold':>10}")
    for c_target in (0.20, 0.40, 0.60, 0.80, 1.00):
        idx = int(c_target * n) - 1
        idx = max(0, min(idx, n - 1))
        print(f"    {cov[idx]:>9.2f}  {err[idx]:>8.3f}  {1 - err[idx]:>8.3f}"
              f"  {conf_sorted[idx]:>10.3f}")

    print(f"\n  AURC (lower = better): {aurc(cov, err):.4f}")

    for r_max in (0.02, 0.05, 0.10):
        c_star = coverage_at_risk(cov, err, r_max)
        print(f"  Max coverage with selective error <= {r_max:.2%}: {c_star:.2%}")

    # Compare with a stronger score: predictive entropy (deep ensembles use this).
    entropy = -np.sum(p * np.log(p + 1e-30), axis=1)
    conf_entropy = 1.0 - entropy / np.log(K)          # 1 = certain, 0 = uniform
    cov2, err2, _ = risk_coverage(y_true, y_pred, conf_entropy)
    print(f"\n  AURC (MSP):      {aurc(cov, err):.4f}")
    print(f"  AURC (entropy):  {aurc(cov2, err2):.4f}"
          "     (either can win depending on classifier calibration)\n")

    print("--- library cross-check (torchsel; selective_classification; cleanlab) ---")
