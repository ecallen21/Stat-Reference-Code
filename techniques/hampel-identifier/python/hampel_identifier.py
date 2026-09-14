"""Hampel identifier / rule (Hampel 1971).

For 1-D data (or a moving window over a signal), flag point i
as an outlier when

    | x_i - median | > k * 1.4826 * MAD

with MAD = median absolute deviation, factor 1.4826 the
Fisher-consistent scale at Gaussian, and threshold k typically
3 (analogous to 3-sigma but robust).

Common time-series application: sliding-window Hampel filter
that replaces outliers with the local median.
"""

import numpy as np    # arrays


def hampel_1d(x, k=3.0):
    """Return boolean outlier mask + robust mean/scale."""
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    scale = 1.4826 * mad
    outlier = np.abs(x - med) > k * scale
    return outlier, med, scale


def hampel_filter(x, window=15, k=3.0):
    """Sliding-window Hampel outlier detector + median replacement."""
    n = len(x)
    x_filtered = x.copy()
    n_out = 0
    for i in range(n):
        lo = max(0, i - window // 2)
        hi = min(n, i + window // 2 + 1)
        seg = x[lo:hi]
        med = np.median(seg)
        mad = np.median(np.abs(seg - med))
        scale = 1.4826 * mad
        if scale > 0 and np.abs(x[i] - med) > k * scale:
            x_filtered[i] = med
            n_out = n_out + 1
    return x_filtered, n_out


def demo():
    print("=== Hampel identifier (Hampel 1971) ===")
    rng = np.random.default_rng(2026)

    print("\n1. Static 1-D: Gaussian sample with wild injected outliers")
    x = rng.standard_normal(200)
    x[[10, 40, 100, 150]] = [15, -20, 25, -10]    # 4 wild outliers
    mask, med, scale = hampel_1d(x, k=3.0)
    detected = np.where(mask)[0]
    print(f"  median = {med:.3f}, robust sigma = {scale:.3f}")
    print(f"  Detected outlier indices : {detected.tolist()}")
    print(f"  Injected indices          : [10, 40, 100, 150]")

    print("\n2. Sliding-window filter on sinusoidal signal with spikes:")
    t = np.linspace(0, 10, 500)
    y_clean = np.sin(t)
    y = y_clean + 0.1 * rng.standard_normal(500)
    spike_idx = rng.choice(500, size=15, replace=False)
    y[spike_idx] = y[spike_idx] + rng.choice([-1, 1], size=15) * (3 + rng.uniform(size=15))

    y_filt, n_replaced = hampel_filter(y, window=15, k=3.0)
    err_before = np.sqrt(np.mean((y - y_clean) ** 2))
    err_after = np.sqrt(np.mean((y_filt - y_clean) ** 2))
    print(f"  Injected 15 spikes; Hampel replaced {n_replaced}")
    print(f"  RMSE before = {err_before:.3f}, after = {err_after:.3f}")


if __name__ == "__main__":
    demo()
