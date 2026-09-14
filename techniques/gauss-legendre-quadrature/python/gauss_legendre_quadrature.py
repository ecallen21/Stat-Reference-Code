"""Gauss-Legendre quadrature (Gauss 1814; Legendre polynomials).

Approximate integral_{-1}^{1} f(x) dx by

    sum_{i=1}^{n} w_i * f(x_i)

where {x_i} are roots of the n-th Legendre polynomial and
{w_i} are the associated weights. n-point Gauss-Legendre is
EXACT for polynomials of degree <= 2n - 1.
"""

import numpy as np    # arrays + linalg


def gauss_legendre_nodes(n):
    """Compute Gauss-Legendre nodes and weights on [-1, 1]
    via the Golub-Welsch algorithm."""
    beta = 0.5 / np.sqrt(1 - (2.0 * np.arange(1, n)) ** (-2))
    J = np.diag(beta, 1) + np.diag(beta, -1)
    w, V = np.linalg.eigh(J)
    x = w
    weights = 2 * V[0, :] ** 2
    return x, weights


def gauss_legendre(f, a, b, n):
    x, w = gauss_legendre_nodes(n)
    scale = (b - a) / 2
    shift = (a + b) / 2
    return scale * np.sum(w * f(scale * x + shift))


def trapezoidal(f, a, b, n):
    x = np.linspace(a, b, n + 1)
    y = f(x)
    return (b - a) / n * (0.5 * y[0] + y[1:-1].sum() + 0.5 * y[-1])


def simpson(f, a, b, n):
    if n % 2 == 1:
        n = n + 1
    x = np.linspace(a, b, n + 1)
    y = f(x)
    return (b - a) / (3 * n) * (y[0] + y[-1] + 4 * y[1:-1:2].sum() + 2 * y[2:-1:2].sum())


def demo():
    print("=== Gauss-Legendre quadrature (Gauss 1814) ===")

    tests = [
        ("polynomial x^5", lambda x: x ** 5, -1, 1, 0.0),
        ("polynomial x^10", lambda x: x ** 10, -1, 1, 2.0 / 11),
        ("Gaussian e^{-x^2}", lambda x: np.exp(-x ** 2), -3, 3, np.sqrt(np.pi) * 0.9999779),
        ("oscillatory sin(20x)", lambda x: np.sin(20 * x), 0, 1, (1 - np.cos(20)) / 20),
    ]
    for label, f, a, b, exact in tests:
        print(f"\n  {label} over [{a}, {b}], exact = {exact:.8f}")
        for n in [4, 8, 16, 32]:
            gl = gauss_legendre(f, a, b, n)
            tz = trapezoidal(f, a, b, n)
            sp = simpson(f, a, b, n)
            print(f"    n={n:2d}  GL={gl:.8f}  err_GL={abs(gl - exact):.2e}   "
                  f"err_trap={abs(tz - exact):.2e}   err_simp={abs(sp - exact):.2e}")


if __name__ == "__main__":
    demo()
