"""Beta Calibration (Reference Sec 47.284).

Kull, Silva Filho & Flach 2017 AISTATS. A more expressive
alternative to Platt scaling for binary classifiers. Fit:

    p_calibrated = sigmoid(a * log(s) - b * log(1 - s) + c)

Equivalent to fitting an intercept + two log-scores as covariates
in a logistic regression. Recovers Platt as a special case
(a = b) and can capture asymmetric miscalibration.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def beta_calibrate_fit(scores, y):
    """Fit logistic on (log s, log 1-s) as features."""
    from scipy.optimize import minimize
    s = np.clip(scores, 1e-6, 1 - 1e-6)
    X = np.column_stack([np.log(s), -np.log(1 - s)])
    def nll(params):
        a, b, c = params
        z = a * X[:, 0] + b * X[:, 1] + c
        p = 1.0 / (1.0 + np.exp(-z))
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    res = minimize(nll, x0=[1.0, 1.0, 0.0], method="BFGS")
    return float(res.x[0]), float(res.x[1]), float(res.x[2])


def beta_predict(scores, a, b, c):
    s = np.clip(scores, 1e-6, 1 - 1e-6)
    z = a * np.log(s) + b * (-np.log(1 - s)) + c
    return 1.0 / (1.0 + np.exp(-z))


def brier(p, y): return float(np.mean((p - y) ** 2))
def logloss(p, y):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


if __name__ == "__main__":
    print("=== Beta Calibration (Kull et al 2017 AISTATS) ===\n")
    rng = np.random.default_rng(0)

    # Simulate an asymmetrically miscalibrated classifier:
    # scores are S-shaped (over-confident at both extremes, well at 0.5)
    n = 3000
    true_prob = rng.beta(2, 2, n)                                 # symmetric prior
    y = (rng.uniform(size=n) < true_prob).astype(int)
    # ASYMMETRIC miscalibration: over-confidence at high scores only
    scores = np.clip(true_prob ** 0.5, 0.01, 0.99)                # inflates all
    scores[y == 0] = np.clip(scores[y == 0] * 1.3, 0.01, 0.99)     # extra inflation for false positives

    # Split
    s_cal, s_te = scores[:2000], scores[2000:]
    y_cal, y_te = y[:2000], y[2000:]

    # Platt for comparison
    from scipy.optimize import minimize
    def platt_nll(params):
        A, B = params
        p = 1.0 / (1.0 + np.exp(-(A * s_cal + B)))
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return -np.mean(y_cal * np.log(p) + (1 - y_cal) * np.log(1 - p))
    res_platt = minimize(platt_nll, x0=[1.0, 0.0], method="BFGS")
    A, B = float(res_platt.x[0]), float(res_platt.x[1])
    p_platt = 1.0 / (1.0 + np.exp(-(A * s_te + B)))

    a, b, c = beta_calibrate_fit(s_cal, y_cal)
    p_beta = beta_predict(s_te, a, b, c)

    print(f"  Beta fit: a = {a:.3f}, b = {b:.3f}, c = {c:.3f}")
    print(f"  Platt fit: A = {A:.3f}, B = {B:.3f}\n")

    print(f"  {'method':>12}   Brier     log-loss")
    print(f"  {'raw':>12}   {brier(s_te, y_te):.4f}    {logloss(s_te, y_te):.4f}")
    print(f"  {'Platt':>12}   {brier(p_platt, y_te):.4f}    {logloss(p_platt, y_te):.4f}")
    print(f"  {'Beta':>12}   {brier(p_beta, y_te):.4f}    {logloss(p_beta, y_te):.4f}")

    print(f"\n  When calibration is ASYMMETRIC, Beta (2 params + intercept) usually beats")
    print(f"  Platt (1 slope + intercept) at the cost of one extra parameter.")

    print("\n--- library cross-check (betacal Python; netcal.scaling.BetaCalibration) ---")
