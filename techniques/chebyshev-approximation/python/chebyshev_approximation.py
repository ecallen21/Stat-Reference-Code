"""Chebyshev polynomial approximation (Chebyshev 1854;
Clenshaw-Curtis 1960).

For smooth f on [-1, 1] the CHEBYSHEV SERIES

    f(x) approx sum_{k=0}^{N} c_k * T_k(x)

converges GEOMETRICALLY (Trefethen 2013). Coefficients via
DCT-II applied to f evaluated at the Chebyshev-Gauss-Lobatto
nodes cos(k pi / N).

Also enables:
    Clenshaw-Curtis quadrature (weights closed-form)
    Barycentric interpolation (numerically stable)
    Near-minimax approximation
"""

import numpy as np    # arrays + fft


def chebyshev_coeffs(f, N):
    """Compute first N+1 Chebyshev coefficients of f on [-1, 1]."""
    x = np.cos(np.pi * np.arange(N + 1) / N)
    y = f(x)
    # DCT-II via FFT (mirror trick)
    ext = np.concatenate([y, y[-2:0:-1]])
    c = np.real(np.fft.fft(ext))[:N + 1] / N
    c[0] = c[0] / 2
    c[N] = c[N] / 2
    return c


def chebyshev_eval(c, x):
    """Clenshaw's recurrence evaluation of a Chebyshev series."""
    N = len(c) - 1
    b_next, b_curr = 0.0, 0.0
    for k in range(N, 0, -1):
        b_next, b_curr = b_curr, 2 * x * b_curr - b_next + c[k]
    return x * b_curr - b_next + c[0]


def demo():
    print("=== Chebyshev approximation (Chebyshev 1854) ===")

    tests = [
        ("smooth 1 / (1 + 25*x^2) (Runge)", lambda x: 1 / (1 + 25 * x ** 2)),
        ("smooth exp(x)", lambda x: np.exp(x)),
        ("piecewise |x|", lambda x: np.abs(x)),
        ("smooth sin(4*pi*x)", lambda x: np.sin(4 * np.pi * x)),
    ]
    test_x = np.linspace(-1, 1, 501)
    for label, f in tests:
        y_true = f(test_x)
        print(f"\n  {label}")
        for N in [10, 20, 40, 80]:
            c = chebyshev_coeffs(f, N)
            y_approx = np.array([chebyshev_eval(c, xi) for xi in test_x])
            err = np.max(np.abs(y_approx - y_true))
            print(f"    N = {N:3d}: sup-norm error = {err:.2e}")


if __name__ == "__main__":
    demo()
