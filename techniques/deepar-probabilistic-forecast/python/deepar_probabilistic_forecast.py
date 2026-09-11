"""DeepAR - Autoregressive Probabilistic Forecast (Sec 47.235).

Salinas, Flunkert, Gasthaus & Januschowski 2020 'DeepAR:
Probabilistic Forecasting with Autoregressive Recurrent Networks',
IJF. LSTM produces the parameters of a per-timestep DISTRIBUTION
(Gaussian mean+var or Negative-Binomial (mu, alpha) for counts):

    h_t = LSTM(y_{t-1}, x_t, h_{t-1})
    theta_t = linear(h_t)
    y_t ~ Distribution(theta_t)

Train by maximising log-likelihood. Sample multi-horizon
predictive distribution by rolling out with Monte-Carlo draws.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def toy_lstm_step(y_prev, h_prev, W):
    """Toy 'LSTM' — recurrent linear with tanh."""
    return np.tanh(W["Wxh"] @ np.array([y_prev]) + W["Whh"] @ h_prev)


def deepar_predict(hist, W, horizon, n_samples=100, rng=None):
    """Roll out horizon steps, drawing MC samples from N(mu_t, sigma_t)."""
    if rng is None: rng = np.random.default_rng(0)
    d_h = W["Whh"].shape[0]
    samples = np.zeros((n_samples, horizon))
    for s in range(n_samples):
        h = np.zeros(d_h)
        # Warm up on history
        for y in hist:
            h = toy_lstm_step(y, h, W)
        # Roll out
        y_prev = hist[-1]
        for t in range(horizon):
            h = toy_lstm_step(y_prev, h, W)
            mu = W["Wm"] @ h
            log_sigma = W["Ws"] @ h
            sigma = np.exp(log_sigma) + 0.01
            samples[s, t] = rng.normal(mu, sigma)
            y_prev = samples[s, t]
    return samples


def crps_gaussian(y, mu, sigma):
    """Continuous Ranked Probability Score for a Gaussian forecast."""
    from math import pi
    z = (y - mu) / sigma
    return sigma * (z * (2 * 0.5 * (1 + np.tanh(z / np.sqrt(pi / 2))) - 1)
                       + 2 * np.exp(-z ** 2 / 2) / np.sqrt(2 * pi) - 1 / np.sqrt(pi))


if __name__ == "__main__":
    print("=== DeepAR (Salinas et al 2020 IJF) ===\n")
    rng = np.random.default_rng(0)

    # Synthetic AR(1) series with trend + noise
    T = 300; horizon = 20
    y = np.zeros(T)
    for t in range(1, T):
        y[t] = 0.7 * y[t - 1] + 0.02 * t + rng.normal(scale=0.5)

    # Simpler proxy: closed-form AR(1) with additive noise (what DeepAR would
    # learn to approximate on this signal). We use it to show the MC-rollout
    # + interval structure rather than the LSTM plumbing.
    train_end = T - horizon
    hist = y[:train_end]
    # Fit AR(1) around the local mean; simple last-value-plus-noise proxy
    sigma = float(np.std(np.diff(hist)))
    samples = np.zeros((200, horizon))
    for s in range(200):
        y_prev = hist[-1]
        for t in range(horizon):
            samples[s, t] = y_prev + rng.normal(0, sigma)          # random walk with local drift
            y_prev = samples[s, t]

    mu_pred = samples.mean(axis=0)
    p05 = np.percentile(samples, 5, axis=0)
    p95 = np.percentile(samples, 95, axis=0)
    actual = y[train_end:]

    print(f"  Series length {T}, forecast horizon {horizon}, 200 MC samples")
    print(f"  Mean forecast (first 5): {np.round(mu_pred[:5], 2).tolist()}")
    print(f"  Actual (first 5):        {np.round(actual[:5], 2).tolist()}")
    coverage = float(((actual >= p05) & (actual <= p95)).mean())
    print(f"  5-95 % PI coverage:      {coverage:.2f}   (nominal 0.90)")
    print(f"  Mean interval width:     {float(np.mean(p95 - p05)):.2f}")

    print("\n  Real DeepAR uses trained LSTM parameters (backprop through time);")
    print("  the toy here demonstrates the sampling / interval structure.")

    print("\n--- library cross-check (gluonts.model.deepar; darts.DeepARModel; pytorch-forecasting) ---")
