"""TSMixer - Time-Series MLP-Mixer (Reference Sec 47.239).

Chen, Ekambaram, Nguyen, Chan, Vahid & Bengio 2023 'TSMixer: An
All-MLP Architecture for Time Series Forecasting', TMLR. Adapts
MLP-Mixer (Tolstikhin 2021) to time series: alternates:

    - TIME-MIXING MLP: (T x C) -> apply MLP across T dim.
    - FEATURE-MIXING MLP: (T x C) -> apply MLP across C dim.

Simple + fast + strong. Beats Transformers on many long-horizon
benchmarks with a fraction of the parameters.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def time_mixing_mlp(X, W_time_hidden, W_time_out):
    """Apply MLP across TIME axis (X shape: T x C)."""
    T, C = X.shape
    # Transpose to (C, T), apply MLP, transpose back
    hidden = np.tanh(X.T @ W_time_hidden)                        # (C, hidden)
    return (hidden @ W_time_out).T                               # (T_out, C)


def feature_mixing_mlp(X, W_feat_hidden, W_feat_out):
    """Apply MLP across FEATURE axis (X shape: T x C)."""
    hidden = np.tanh(X @ W_feat_hidden)                          # (T, hidden)
    return hidden @ W_feat_out                                    # (T, C_out)


def tsmixer_forecast(input_series, forecast_horizon, W_time_hidden, W_time_out,
                       W_feat_hidden, W_feat_out):
    """Two mixer blocks: time-mix -> feature-mix."""
    x = time_mixing_mlp(input_series, W_time_hidden, W_time_out)
    x = feature_mixing_mlp(x, W_feat_hidden, W_feat_out)
    return x[-forecast_horizon:]                                 # last H rows are the forecast


if __name__ == "__main__":
    print("=== TSMixer (Chen et al 2023 TMLR) ===\n")
    rng = np.random.default_rng(0)

    T = 100; horizon = 20; C = 3                                 # 3 correlated series
    t = np.arange(T + horizon)
    series = np.stack([
        3 * np.sin(2 * np.pi * t / 24) + 0.5 * rng.normal(size=T + horizon),
        2 * np.cos(2 * np.pi * t / 24) + 0.5 * rng.normal(size=T + horizon),
        0.05 * t + 0.5 * rng.normal(size=T + horizon),
    ], axis=1)                                                   # shape (T + horizon, C)

    # Weights fit via ridge regression on the training window
    hidden_dim = 32
    W_time_hidden = rng.normal(scale=0.1, size=(T, hidden_dim))
    W_feat_hidden = rng.normal(scale=0.3, size=(C, hidden_dim))
    W_feat_out = np.eye(hidden_dim)[:C].T                        # identity-like
    # Fit W_time_out via ridge: (hidden -> T+horizon)
    hidden = np.tanh(series[:T].T @ W_time_hidden)               # (C, hidden_dim)
    target = series.T                                             # (C, T + horizon)
    W_time_out = np.linalg.lstsq(hidden, target, rcond=None)[0]
    forecast = tsmixer_forecast(series[:T], horizon, W_time_hidden, W_time_out,
                                     W_feat_hidden, W_feat_out)
    actual = series[T:T + horizon]
    err_ls = float(np.mean((forecast - actual) ** 2))
    naive_baseline = np.tile(series[T - 1], (horizon, 1))
    err_naive = float(np.mean((naive_baseline - actual) ** 2))
    print(f"  Multi-series input: T={T}, C={C}, horizon={horizon}")
    print(f"  LS-fit TSMixer forecast MSE:      {err_ls:.3f}")
    print(f"  Naive last-value MSE:             {err_naive:.3f}")

    print(f"\n  TSMixer alternates time-mixing MLP (across T) with feature-mixing")
    print(f"  MLP (across C). All-MLP: fast, simple, no attention, no recurrence.")

    print("\n--- library cross-check (google-research/google-research/tree/master/tsmixer; darts) ---")
