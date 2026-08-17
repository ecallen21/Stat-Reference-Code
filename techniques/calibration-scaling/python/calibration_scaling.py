"""Probability calibration + reliability + Platt / isotonic scaling
(Reference §26.15).

A classifier that predicts p(y = 1 | x) = 0.9 is well-calibrated iff, among
all inputs where it predicts 0.9, roughly 90% truly have y = 1.

Diagnostics
    Reliability diagram: bin predictions by predicted probability; plot mean
    predicted vs empirical positive rate.  Perfectly-calibrated = diagonal.
    Brier score = mean((p - y)^2)
    Expected calibration error (ECE) = weighted mean bin-wise miscalibration
    Log loss = -mean(y log p + (1 - y) log(1 - p))

Post-hoc calibration (on a HELD-OUT CALIBRATION SET)
    Platt scaling (Platt 1999): logistic sigmoid  p_cal = 1 / (1 + exp(-a s - b))
    Isotonic regression: nonparametric monotone map (see 'isotonic-regression')
    Temperature scaling: single scalar T for neural nets, softmax(z / T)
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy.optimize import minimize    # SciPy optimizer (BFGS/Newton) for MLE


def brier_score(p, y):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def log_loss(p, y):
    p = np.clip(np.asarray(p), 1e-12, 1 - 1e-12); y = np.asarray(y)
    return -float(np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def reliability_diagram(p, y, n_bins: int = 10) -> dict:
    p = np.asarray(p); y = np.asarray(y)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(p, bins) - 1
    idx = np.clip(idx, 0, n_bins - 1)
    means_pred = []; means_true = []; sizes = []
    for k in range(n_bins):
        mask = idx == k
        if mask.sum() == 0: continue
        means_pred.append(float(p[mask].mean()))
        means_true.append(float(y[mask].mean()))
        sizes.append(int(mask.sum()))
    means_pred = np.array(means_pred); means_true = np.array(means_true); sizes = np.array(sizes)
    ece = float(np.sum(sizes * np.abs(means_pred - means_true)) / sizes.sum())
    return {"means_pred": means_pred, "means_true": means_true, "bin_sizes": sizes,
            "expected_calibration_error": ece}


def platt_scaling(s, y):
    """Fit p = sigmoid(a * s + b) on calibration set."""
    s = np.asarray(s, dtype=float); y = np.asarray(y, dtype=float)
    def neg_ll(theta):
        a, b = theta
        z = a * s + b
        return -np.sum(y * z - np.logaddexp(0, z))
    res = minimize(neg_ll, [1.0, 0.0], method="BFGS")
    a, b = res.x
    def predict(s_new): return 1 / (1 + np.exp(-(a * np.asarray(s_new) + b)))
    return {"a": float(a), "b": float(b), "predict": predict, "method": "Platt scaling"}


def isotonic_scaling(s, y):
    """Fit monotone step function via Pool-Adjacent-Violators."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "isotonic-regression", "python"))
    from isotonic_regression import pava
    s = np.asarray(s, dtype=float); y = np.asarray(y, dtype=float)
    order = np.argsort(s); s_sort = s[order]; y_sort = y[order]
    y_iso = pava(y_sort, increasing=True)
    def predict(s_new):
        s_new = np.asarray(s_new, dtype=float)
        return np.interp(s_new, s_sort, y_iso, left=y_iso[0], right=y_iso[-1])
    return {"predict": predict, "method": "Isotonic scaling"}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # Simulate over-confident classifier scores
    n = 2000
    y = rng.binomial(1, 0.4, n)
    # Score correlates with y but is overconfident (pushed to 0/1)
    raw = 3 * (y - 0.5) + rng.normal(0, 1, n)
    p_over = 1 / (1 + np.exp(-3 * raw))          # over-scale factor

    # Split half calibration, half test
    perm = rng.permutation(n)
    cal = perm[:n // 2]; te = perm[n // 2:]

    print(f"=== Before calibration ===")
    print(f"  Brier = {brier_score(p_over[te], y[te]):.4f}")
    print(f"  LogLoss = {log_loss(p_over[te], y[te]):.4f}")
    print(f"  ECE = {reliability_diagram(p_over[te], y[te])['expected_calibration_error']:.4f}")

    print(f"\n=== Platt scaling ===")
    plat = platt_scaling(raw[cal], y[cal])
    p_plat = plat["predict"](raw[te])
    print(f"  a = {plat['a']:.3f}, b = {plat['b']:.3f}")
    print(f"  Brier = {brier_score(p_plat, y[te]):.4f}")
    print(f"  LogLoss = {log_loss(p_plat, y[te]):.4f}")
    print(f"  ECE = {reliability_diagram(p_plat, y[te])['expected_calibration_error']:.4f}")

    print(f"\n=== Isotonic scaling ===")
    iso = isotonic_scaling(raw[cal], y[cal])
    p_iso = iso["predict"](raw[te])
    print(f"  Brier = {brier_score(p_iso, y[te]):.4f}")
    print(f"  LogLoss = {log_loss(p_iso, y[te]):.4f}")
    print(f"  ECE = {reliability_diagram(p_iso, y[te])['expected_calibration_error']:.4f}")
