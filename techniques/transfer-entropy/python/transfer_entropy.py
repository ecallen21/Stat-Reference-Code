"""Transfer entropy (Reference Sec 34.11).

Schreiber (2000) 'Measuring information transfer.'

Quantifies the DIRECTED information flow from process X to Y:

  TE(X -> Y) = sum p(y_{t+1}, y_t^(k), x_t^(l))
                    * log ( p(y_{t+1} | y_t^(k), x_t^(l))
                            / p(y_{t+1} | y_t^(k)) )
             = H(y_{t+1} | y_t^(k))  -  H(y_{t+1} | y_t^(k), x_t^(l)).

Non-zero TE means X's history reduces uncertainty about Y_{t+1} BEYOND
Y's own history -- Granger causality's non-linear cousin.

Here we implement discretised TE with lag-1 history on synthetic
coupled processes and confirm TE(X->Y) > 0 when Y truly depends on X.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def _entropy_joint(*cols):
    """Entropy of the joint distribution of a set of discrete columns."""
    n = len(cols[0])
    stacked = np.stack(cols, axis=1)
    _, counts = np.unique(stacked, axis=0, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log(p)).sum())


def transfer_entropy(x, y, lag=1):
    """TE(X -> Y) using lag-1 history on discretised series."""
    # y_next = y_{t+1}, y_hist = y_t, x_hist = x_t; discard first `lag` samples.
    y_next = y[lag:]
    y_hist = y[:-lag]
    x_hist = x[:-lag]
    # H(y_next | y_hist) - H(y_next | y_hist, x_hist)
    H_ynext_yhist = _entropy_joint(y_next, y_hist) - _entropy_joint(y_hist)
    H_ynext_yhist_xhist = (_entropy_joint(y_next, y_hist, x_hist)
                             - _entropy_joint(y_hist, x_hist))
    return max(0.0, H_ynext_yhist - H_ynext_yhist_xhist)


def _discretise(x, n_bins=4):
    edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    return np.clip(np.searchsorted(edges[1:-1], x), 0, n_bins - 1)


if __name__ == "__main__":
    print("=== Transfer entropy (Schreiber 2000) ===\n")
    rng = np.random.default_rng(0)
    T = 2000

    # Scenario A: Y_{t+1} depends on X_t (X drives Y)
    x = rng.normal(0, 1, T)
    y = np.zeros(T)
    for t in range(1, T):
        y[t] = 0.5 * y[t - 1] + 0.7 * x[t - 1] + rng.normal(0, 0.3)

    xd = _discretise(x); yd = _discretise(y)
    te_xy = transfer_entropy(xd, yd)
    te_yx = transfer_entropy(yd, xd)
    print(f"  X drives Y:")
    print(f"    TE(X -> Y) = {te_xy:.4f} nats   (should be > 0)")
    print(f"    TE(Y -> X) = {te_yx:.4f} nats   (should be near 0)\n")

    # Scenario B: X and Y independent
    xi = rng.normal(0, 1, T)
    yi = rng.normal(0, 1, T)
    te_xy_ind = transfer_entropy(_discretise(xi), _discretise(yi))
    te_yx_ind = transfer_entropy(_discretise(yi), _discretise(xi))
    print(f"  Independent:")
    print(f"    TE(X -> Y) = {te_xy_ind:.4f} nats   (should be near 0)")
    print(f"    TE(Y -> X) = {te_yx_ind:.4f} nats   (should be near 0)\n")

    print("--- library cross-check (JIDT (Lizier); R RTransferEntropy; Python PyIF) ---")
