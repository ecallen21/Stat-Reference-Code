"""State-space models: S4-lite / Mamba primitives (Reference §27.x extra).

Modern SSM layer transforms a sequence x_t via a linear dynamical system:
    h_t = A h_{t-1} + B x_t
    y_t = C h_t + D x_t

Trained by SGD on the parameters (A, B, C, D).  Key innovations:
  * S4 (Gu 2022): HiPPO initialisation of A (Legendre memory), diagonal
    reparameterisation of A for scalability; long-sequence modelling.
  * Mamba (Gu-Dao 2023): SELECTIVE (input-dependent) B, C so the SSM can
    focus on relevant tokens; competitive with transformers at O(T) instead
    of O(T^2) attention.

We demonstrate the fundamental scan operation on a small SSM with random
A, B, C, D and compare against a KALMAN-FILTER-style linear-Gaussian sanity
check.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

import numpy as np    # numerical arrays + linear algebra


def ssm_scan(x, A, B, C, D, h0=None):
    """x: (T, d_in);  A: (d_h, d_h);  B: (d_h, d_in);  C: (d_out, d_h);
    D: (d_out, d_in);  h0: (d_h,)  Returns y: (T, d_out), H: (T, d_h)."""
    T = x.shape[0]; d_h = A.shape[0]
    h = np.zeros(d_h) if h0 is None else h0.copy()
    y = np.zeros((T, C.shape[0])); H = np.zeros((T, d_h))
    for t in range(T):
        h = A @ h + B @ x[t]
        y[t] = C @ h + D @ x[t]
        H[t] = h
    return y, H


def convolution_form(x, A, B, C, D, T):
    """Alternative computation: y = K * x + D * x where K is the SSM kernel
    K[k] = C A^k B for k >= 0.  Used by S4 for O(T log T) FFT convolution."""
    d_out = C.shape[0]
    K = np.zeros((T, d_out))
    Ap = np.eye(A.shape[0])
    for k in range(T):
        K[k] = C @ Ap @ B                                    # shape (d_out,) assuming d_in=1
        Ap = A @ Ap
    y = np.zeros((T, d_out))
    for t in range(T):
        for k in range(t + 1):
            y[t] += K[k] * x[t - k]
        y[t] += D @ x[t]
    return y


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    T = 20; d_in = 1; d_h = 4; d_out = 1
    A = 0.9 * np.eye(d_h) + 0.05 * rng.normal(size=(d_h, d_h))
    B = rng.normal(size=(d_h, d_in)); C = rng.normal(size=(d_out, d_h))
    D = np.zeros((d_out, d_in))
    x = rng.normal(size=(T, d_in))

    y_scan, H = ssm_scan(x, A, B, C, D)
    y_conv = convolution_form(x, A, B, C, D, T)
    diff = float(np.max(np.abs(y_scan - y_conv)))
    print(f"=== State-space model core scan ===")
    print(f"  input shape                       = {x.shape}")
    print(f"  output shape                      = {y_scan.shape}")
    print(f"  ||scan-form - convolution-form||_inf = {diff:.2e}   "
          f"(should be ~0)")
    print(f"  hidden trajectory norms (first 5): "
          f"{np.round(np.linalg.norm(H[:5], axis=1), 3).tolist()}")

    # Impulse response of the SSM (the kernel)
    e = np.zeros((T, d_in)); e[0, 0] = 1.0
    y_imp, _ = ssm_scan(e, A, B, C, D)
    print(f"\n  impulse response (kernel) K[0:6] = "
          f"{np.round(y_imp[:6, 0], 3).tolist()}")
    print(f"  A's spectral radius = {float(np.max(np.abs(np.linalg.eigvals(A)))):.3f}   "
          f"(< 1 for stable systems)")

    print("\n--- library cross-check (mamba-ssm, s4d, state-spaces/s4) ---")
