"""Mamba / structured state-space models (S4, S5) (Reference Sec 47.22).

Gu, Goel & Re 2022 'S4: efficiently modeling long sequences with
structured state spaces', ICLR; Gu & Dao 2023 'Mamba: linear-time
sequence modeling with selective state spaces', arXiv. A transformer
alternative built from LINEAR TIME-INVARIANT / SELECTIVE state-space
recurrences that admit both parallel (convolutional) training and
linear-time inference.

Core discrete SSM:

    h_t = A * h_{t-1} + B * x_t
    y_t = C * h_t     (+ D * x_t)

  * S4: A, B, C, D fixed across the sequence; efficient via HiPPO
    initialisation + Cauchy kernel FFT.
  * Mamba: SELECTIVE (input-dependent) B and C; scan-based training
    with O(L * d) work and O(1) state per position.

For didactic purposes we implement a compact FIXED-A SSM (S4-style
without HiPPO tricks) and demonstrate: forward pass, convolutional
equivalence via kernel unrolling, and stable long-context memory.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def discrete_ssm_forward(A, B, C, D, x):
    """Runs y = SSM(x) using the recurrence."""
    L, d_in = x.shape
    n = A.shape[0]
    h = np.zeros(n)
    y = np.zeros(L)
    for t in range(L):
        h = A @ h + B * x[t, 0]
        y[t] = C @ h + D * x[t, 0]
    return y


def ssm_kernel(A, B, C, D, L):
    """Compute the impulse-response kernel k[0..L-1] such that y = conv1d(k, x)."""
    n = A.shape[0]
    k = np.zeros(L)
    #  k[0] = D
    k[0] = D
    #  A^t B  for t = 0, 1, ..., L-1;  y = sum_t C A^t B x_{ct-t}
    A_t = np.eye(n)
    for t in range(L):
        contrib = C @ A_t @ B
        if t == 0:
            k[t] += contrib
        else:
            k[t] = contrib
        A_t = A_t @ A
    return k


def conv1d(k, x):
    L = len(x)
    y = np.zeros(L)
    for t in range(L):
        for s in range(t + 1):
            y[t] += k[s] * x[t - s]
    return y


def hippo_init(n):
    """Simple stable diagonal 'A' -- eigenvalues in (0, 1)."""
    return np.diag(np.linspace(0.99, 0.7, n))


if __name__ == "__main__":
    print("=== Structured state-space models (S4 / Mamba flavour) ===\n")
    rng = np.random.default_rng(0)
    n = 8
    A = hippo_init(n)
    B = rng.normal(size=n) * 0.5
    C = rng.normal(size=n) * 0.5
    D = 0.1

    L = 40
    x = np.zeros((L, 1))
    x[5, 0] = 1.0   # impulse at t=5

    y_rec = discrete_ssm_forward(A, B, C, D, x)
    k = ssm_kernel(A, B, C, D, L)
    y_conv = conv1d(k, x[:, 0])

    max_err = np.max(np.abs(y_rec - y_conv))
    print(f"  Recurrent vs convolutional forward pass: max |diff| = {max_err:.3e}")
    print(f"  (Both should agree exactly for linear time-invariant SSM.)")

    #  Long-context stability check
    L_long = 500
    x_long = rng.normal(size=(L_long, 1)) * 0.3
    y_long = discrete_ssm_forward(A, B, C, D, x_long)
    print(f"\n  L = {L_long} steps:  |y|_max = {np.abs(y_long).max():.3f}  "
          f"(bounded thanks to |eig(A)| < 1)")

    #  Show kernel decay
    kernel_energy_first_20 = float(np.sum(k[:20] ** 2) / np.sum(k ** 2))
    print(f"  Kernel energy in first 20 taps = {kernel_energy_first_20:.3f}  "
          f"(effective receptive field)")

    print("\n--- library cross-check (state-spaces / mamba-ssm Python; no R) ---")
