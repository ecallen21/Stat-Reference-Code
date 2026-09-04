"""Circular statistics (Reference Sec 38.3).

Angles / directions live on a circle -- 0 and 2*pi are the SAME point,
so the arithmetic mean of 350 deg and 10 deg is NOT 180.  Mardia-Jupp
2000: build statistics from vector-mean representation of the data.

  Mean direction   : theta_bar = atan2(sum sin, sum cos)
  Mean resultant   : R_bar = sqrt((sum sin)^2 + (sum cos)^2) / n
  Circular variance: V = 1 - R_bar   (0 = concentrated, 1 = uniform)
  Concentration    : kappa (von Mises MLE via Bessel-ratio inverse)

  Rayleigh test    : H_0 uniform vs H_1 unimodal   -- test stat n*R_bar^2

Von Mises pdf: p(theta | mu, kappa) = exp(kappa cos(theta - mu))
                                       / (2 pi I_0(kappa))
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays
from scipy import special, optimize    # Bessel functions for von Mises kappa


def circular_mean(theta):
    """Mean direction (radians) and mean resultant length."""
    C = np.cos(theta).mean()
    S = np.sin(theta).mean()
    R = np.sqrt(C ** 2 + S ** 2)
    mu = np.arctan2(S, C)
    return {"mean_direction": float(mu), "R_bar": float(R),
            "circular_var": float(1 - R)}


def rayleigh_test(theta):
    """Rayleigh test of uniformity (H_0: uniform on circle)."""
    n = len(theta)
    R = np.sqrt(np.cos(theta).sum() ** 2 + np.sin(theta).sum() ** 2) / n
    Z = n * R ** 2
    # Approximate p-value (Fisher 1993 correction):
    p = np.exp(-Z) * (1 + (2 * Z - Z ** 2) / (4 * n) -
                     (24 * Z - 132 * Z ** 2 + 76 * Z ** 3 - 9 * Z ** 4) / (288 * n ** 2))
    return {"R_bar": float(R), "Z": float(Z), "p_value": float(p)}


def vonmises_kappa_mle(theta):
    """MLE for the von Mises concentration kappa given mean direction estimate."""
    R = np.sqrt(np.cos(theta).sum() ** 2 + np.sin(theta).sum() ** 2) / len(theta)
    # invert Bessel-ratio I1/I0 = R
    obj = lambda k: (special.i1(k) / special.i0(k) - R) ** 2
    kappa = optimize.minimize_scalar(obj, bounds=(1e-6, 200), method="bounded").x
    return float(kappa)


if __name__ == "__main__":
    print("=== Circular statistics: mean direction, Rayleigh, von Mises kappa ===\n")
    rng = np.random.default_rng(0)
    # 60 wind-direction measurements from a von Mises around due-north
    n = 60
    mu_true, kappa_true = np.deg2rad(10), 4.0
    theta = rng.vonmises(mu=mu_true, kappa=kappa_true, size=n)
    theta = np.mod(theta, 2 * np.pi)

    m = circular_mean(theta)
    print(f"  Sample n = {n}   true mu = 10 deg   true kappa = {kappa_true}")
    print(f"  Mean direction  = {np.rad2deg(m['mean_direction']):.2f} deg")
    print(f"  Mean resultant  = {m['R_bar']:.3f}")
    print(f"  Circular var    = {m['circular_var']:.3f}")

    r = rayleigh_test(theta)
    print(f"\n  Rayleigh test  Z = {r['Z']:.2f}   p = {r['p_value']:.3e}"
          f"   ({'reject' if r['p_value'] < 0.05 else 'fail-to-reject'} uniformity)")

    khat = vonmises_kappa_mle(theta)
    print(f"\n  von Mises kappa MLE = {khat:.2f}   (true {kappa_true})")

    # Contrast: naive arithmetic mean of degrees ignoring wrap-around
    theta_deg = np.rad2deg(theta)
    arith = theta_deg.mean()
    print(f"\n  Naive mean of degrees ignoring wrap = {arith:.2f} deg   -- WRONG.\n")

    print("--- library cross-check (R circular::mle.vonmises/rayleigh.test; Python custom) ---")
