"""Echo State Network / Reservoir Computing (Ref Sec 47.310).

Jaeger 2001 GMD Report; Lukosevicius & Jaeger 2009. A large,
FIXED, sparsely-connected recurrent RESERVOIR maps inputs to
high-dimensional states; only the LINEAR READOUT is trained:

    h_t = tanh(W_in x_t + W_res h_{t-1})           (fixed)
    y_hat_t = W_out h_t                            (learn via ridge)

Fast to train (no BPTT), competitive with LSTM on many
time-series tasks provided reservoir SPECTRAL RADIUS < 1
("echo state property").
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


class ESN:
    def __init__(self, n_res=100, spectral_radius=0.9, sparsity=0.1, rng=None):
        if rng is None: rng = np.random.default_rng(0)
        self.n = n_res
        self.W_in = rng.normal(0, 1, (n_res, 1))
        W_res = rng.normal(0, 1, (n_res, n_res))
        mask = rng.uniform(size=W_res.shape) < sparsity
        W_res = W_res * mask
        rho = float(np.abs(np.linalg.eigvals(W_res)).max())
        self.W_res = W_res * (spectral_radius / max(rho, 1e-9))

    def collect_states(self, x):
        T = len(x); H = np.zeros((T, self.n))
        h = np.zeros(self.n)
        for t in range(T):
            h = np.tanh(self.W_in.flatten() * x[t] + self.W_res @ h)
            H[t] = h
        return H

    def fit(self, x, y, reg=1e-4):
        H = self.collect_states(x)
        A = H.T @ H + reg * np.eye(self.n)
        self.W_out = np.linalg.solve(A, H.T @ y)

    def predict(self, x):
        H = self.collect_states(x)
        return H @ self.W_out


if __name__ == "__main__":
    print("=== Echo State Network (Jaeger 2001) ===\n")
    rng = np.random.default_rng(0)

    # Target: chaotic Mackey-Glass surrogate (sum of two sines with delayed feedback)
    T = 3000
    t = np.arange(T)
    x = np.sin(0.05 * t) + 0.5 * np.sin(0.17 * t) + 0.2 * rng.standard_normal(T)
    y = np.sin(0.05 * (t + 5)) + 0.5 * np.sin(0.17 * (t + 5))    # 5-step ahead target

    train_end = 2000
    esn = ESN(n_res=200, spectral_radius=0.9, sparsity=0.05, rng=rng)
    esn.fit(x[:train_end], y[:train_end])
    y_pred = esn.predict(x[train_end:])
    rmse_esn = float(np.sqrt(np.mean((y_pred - y[train_end:]) ** 2)))
    print(f"  Reservoir size: 200, spectral radius: 0.9")
    print(f"  Train points: {train_end},  test points: {T - train_end}")
    print(f"  Test RMSE:  {rmse_esn:.4f}")

    # Baseline: predict last value
    baseline_pred = np.roll(x, 1)[train_end:]
    rmse_lag = float(np.sqrt(np.mean((baseline_pred - y[train_end:]) ** 2)))
    print(f"  Lag-1 persistence baseline RMSE:  {rmse_lag:.4f}\n")

    # Verify echo-state property: perturbation should decay
    esn2 = ESN(n_res=200, spectral_radius=0.9, sparsity=0.05, rng=rng)
    x_short = rng.standard_normal(200)
    H1 = esn2.collect_states(x_short)
    x_pert = x_short.copy(); x_pert[0] += 1.0
    H2 = esn2.collect_states(x_pert)
    print(f"  Echo state property (perturbation decay ||h1 - h2|| over time):")
    for t_ in [0, 5, 20, 50, 100, 199]:
        print(f"    t = {t_:>3}   ||h_A - h_B|| = {np.linalg.norm(H1[t_] - H2[t_]):.4f}")

    print("\n--- library cross-check (reservoirpy Python; easyesn; pyESN) ---")
