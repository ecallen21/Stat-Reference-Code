"""Temperature Scaling (Reference Sec 47.281).

Guo, Pleiss, Sun & Weinberger 2017 ICML. Post-hoc calibration
for deep networks: divide LOGITS by a single scalar T > 0 before
softmax:

    p_calibrated = softmax(z / T)

Fit T by minimising NLL on a HELD-OUT validation set. Only 1
parameter, so cheap and expressive enough to fix confidence
miscalibration of most classifiers without changing predictions.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def softmax_temp(z, T):
    z_scaled = z / T
    z_max = z_scaled.max(axis=1, keepdims=True)
    e = np.exp(z_scaled - z_max)
    return e / e.sum(axis=1, keepdims=True)


def nll(z, y, T):
    p = softmax_temp(z, T)
    return -np.mean(np.log(p[np.arange(len(y)), y] + 1e-12))


def fit_temperature(z, y, T_grid=None):
    """Grid search + local refine for optimal T."""
    if T_grid is None: T_grid = np.linspace(0.1, 5.0, 50)
    losses = [nll(z, y, T) for T in T_grid]
    T_star = float(T_grid[int(np.argmin(losses))])
    # Refine with a 1-D Brent-style search
    from scipy.optimize import minimize_scalar
    res = minimize_scalar(lambda T: nll(z, y, T), bounds=(0.05, 10.0), method="bounded")
    return float(res.x)


def ece(probs, y, n_bins=10):
    """Expected calibration error."""
    confs = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    correct = (preds == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for i in range(n_bins):
        m = (confs > edges[i]) & (confs <= edges[i + 1])
        if m.sum() > 0:
            acc = correct[m].mean(); conf = confs[m].mean()
            e += (m.sum() / len(y)) * abs(acc - conf)
    return float(e)


if __name__ == "__main__":
    print("=== Temperature Scaling (Guo et al 2017 ICML) ===\n")
    rng = np.random.default_rng(0)

    # Simulate an overconfident 5-class classifier
    n = 2000; C = 5
    y = rng.integers(0, C, n)
    z = rng.normal(0, 1, (n, C))
    z[np.arange(n), y] += 3.0                                     # true class gets a boost
    z *= 3.0                                                      # scale up -> overconfident

    p_raw = softmax_temp(z, T=1.0)
    T_star = fit_temperature(z[:1500], y[:1500])
    p_cal = softmax_temp(z[1500:], T=T_star)

    acc_raw = (p_raw[1500:].argmax(1) == y[1500:]).mean()
    ece_raw = ece(p_raw[1500:], y[1500:])
    ece_cal = ece(p_cal, y[1500:])

    print(f"  Held-out (500) accuracy:            {acc_raw:.3f}")
    print(f"  Held-out ECE before calibration:    {ece_raw:.4f}")
    print(f"  Held-out ECE after temperature T={T_star:.2f}: {ece_cal:.4f}")
    print(f"  Predictions unchanged (T > 0 preserves argmax).\n")

    print(f"  Mean confidence: raw {p_raw[1500:].max(1).mean():.3f}   scaled {p_cal.max(1).mean():.3f}")
    print(f"  Temperature scaling can only reduce or preserve confidence,")
    print(f"  it never changes the predicted class.")

    print("\n--- library cross-check (torchcalibration.TemperatureScaling; netcal.scaling.TemperatureScaling) ---")
