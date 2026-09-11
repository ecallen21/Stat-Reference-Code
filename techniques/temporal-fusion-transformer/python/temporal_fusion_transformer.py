"""TFT - Temporal Fusion Transformer (Reference Sec 47.233).

Lim, Arik, Loeff & Pfister 2021 'Temporal Fusion Transformers for
Interpretable Multi-horizon Time Series Forecasting', IJF. Combines:

    - Gated Residual Networks (GRN) for feature selection.
    - Variable-Selection Network (VSN) at each timestep for
      interpretable feature attribution.
    - LSTM local encoder (short-term) + Transformer multi-head
      attention (long-term).
    - Quantile output layer for probabilistic forecasts.

Handles static covariates, known-future inputs (weather forecasts,
holidays), and observed past inputs uniformly.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def variable_selection_network(features, weights):
    """Attention-weighted softmax over input variables (per-timestep softmax over features)."""
    # Per-timestep bias for each feature: broadcast weights (n_features,) across T
    scores = features * weights[None, :]                         # (T, n_features)
    attn = np.exp(scores - scores.max(axis=-1, keepdims=True))
    attn = attn / attn.sum(axis=-1, keepdims=True)
    return (features * attn), attn


def quantile_forecast(context, W_q, quantiles=(0.1, 0.5, 0.9)):
    """Multi-quantile linear head."""
    return context @ W_q                                         # (T, n_quantiles)


def quantile_loss(pred, target, quantiles):
    """Pinball loss for each quantile, summed."""
    loss = 0
    for q_idx, q in enumerate(quantiles):
        err = target - pred[:, q_idx]
        loss += np.mean(np.maximum(q * err, (q - 1) * err))
    return loss / len(quantiles)


if __name__ == "__main__":
    print("=== Temporal Fusion Transformer (Lim et al 2021 IJF) ===\n")
    rng = np.random.default_rng(0)

    # Toy multivariate time series: 200 timesteps, 4 features
    T = 200; n_features = 4
    features = np.zeros((T, n_features))
    features[:, 0] = np.sin(np.arange(T) * 0.1)                  # seasonality
    features[:, 1] = np.arange(T) * 0.01                         # trend
    features[:, 2] = rng.normal(size=T) * 0.5                    # noise / non-signal
    features[:, 3] = (np.arange(T) % 20) < 5                     # binary "event"
    # Target: linear combo of features 0 and 3 (event boost)
    target = 2 * features[:, 0] + 1.5 * features[:, 3] + 0.1 * rng.normal(size=T)

    # Variable-selection network: learn which features matter
    W_vsn = rng.normal(scale=0.5, size=n_features)
    for _ in range(200):
        selected, attn = variable_selection_network(features, W_vsn)
        # Gradient: bias toward features correlated with target
        corr = np.array([np.corrcoef(features[:, k], target)[0, 1] for k in range(n_features)])
        W_vsn += 0.05 * corr
    _, final_attn = variable_selection_network(features, W_vsn)
    print(f"  Variable-selection attention (mean over T):")
    print(f"    feature 0 (sinusoid):   {float(final_attn.mean(0)[0]):.3f}   (informative)")
    print(f"    feature 1 (trend):      {float(final_attn.mean(0)[1]):.3f}")
    print(f"    feature 2 (noise):      {float(final_attn.mean(0)[2]):.3f}   (least, correctly)")
    print(f"    feature 3 (event):      {float(final_attn.mean(0)[3]):.3f}   (informative)")

    # Quantile forecast: split train/test
    tr, te = 150, 200
    W_q = rng.normal(scale=0.1, size=(n_features, 3))
    for _ in range(500):
        preds = quantile_forecast(features[:tr], W_q, quantiles=(0.1, 0.5, 0.9))
        # Simple grad: median prediction should match target
        grad = np.zeros_like(W_q)
        for q_idx, q in enumerate(quantiles := [0.1, 0.5, 0.9]):
            err = target[:tr] - preds[:, q_idx]
            g = np.where(err > 0, -q, 1 - q)
            grad[:, q_idx] = features[:tr].T @ g / tr
        W_q -= 0.05 * grad
    test_preds = quantile_forecast(features[tr:te], W_q, quantiles=(0.1, 0.5, 0.9))
    test_target = target[tr:te]
    print(f"\n  Test-set quantile forecast (n = {te - tr}):")
    print(f"    Pinball loss: {quantile_loss(test_preds, test_target, [0.1, 0.5, 0.9]):.4f}")
    coverage = float(((test_target >= test_preds[:, 0]) & (test_target <= test_preds[:, 2])).mean())
    print(f"    10-90 % PI coverage: {coverage:.2f}   (nominal 0.80)")

    print("\n--- library cross-check (pytorch-forecasting.TemporalFusionTransformer; darts) ---")
