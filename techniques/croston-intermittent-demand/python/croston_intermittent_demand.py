"""Croston's method for intermittent-demand forecasting
(Croston 1972; Syntetos-Boylan-Croston 2005 correction).

Split the series into non-zero demand sizes z_t and inter-
arrival times x_t; forecast each with SES:

    z_hat_{t+1} = alpha z_t + (1 - alpha) z_hat_t
    x_hat_{t+1} = alpha x_t + (1 - alpha) x_hat_t

Forecast rate = z_hat / x_hat. Syntetos-Boylan-Croston (SBC)
multiply by (1 - alpha/2) to remove the bias in Croston's
estimator.
"""

import numpy as np    # arrays


def croston(y, alpha=0.1, method="classic"):
    n = len(y)
    nz_idx = np.where(y > 0)[0]
    if len(nz_idx) < 2:
        return np.full(n, y.mean())
    z, x = [], []
    prev_nz = -1
    for i in nz_idx:
        z.append(y[i])
        x.append(i - prev_nz)
        prev_nz = i
    z, x = np.array(z, dtype=float), np.array(x, dtype=float)
    # SES on z and x
    z_hat, x_hat = np.empty_like(z), np.empty_like(x)
    z_hat[0], x_hat[0] = z[0], x[0]
    for t in range(1, len(z)):
        z_hat[t] = alpha * z[t] + (1 - alpha) * z_hat[t - 1]
        x_hat[t] = alpha * x[t] + (1 - alpha) * x_hat[t - 1]
    rate = z_hat[-1] / x_hat[-1]
    if method == "sbc":
        rate = rate * (1 - alpha / 2)
    return rate


def demo():
    print("=== Croston / SBC (Croston 1972; Syntetos-Boylan 2005) ===")
    rng = np.random.default_rng(2026)
    n = 60
    # true expected rate ~ (0.3 prob) * (mean size 10) = 3.0 / period
    prob, mean_size = 0.3, 10
    y = rng.binomial(1, prob, size=n) * rng.poisson(mean_size, size=n)
    print(f"  True mean rate ~ {prob * mean_size:.2f} per period")
    print(f"  Empirical  mean = {y.mean():.2f}")

    for alpha in [0.1, 0.3]:
        for method in ["classic", "sbc"]:
            r = croston(y, alpha=alpha, method=method)
            print(f"  alpha={alpha}, method={method:7s}: forecast rate = {r:.3f}")

    print(f"\n  First 20 obs of demand : {y[:20].tolist()}")


if __name__ == "__main__":
    demo()
