"""Item + test information functions (Reference §22.14).

The Fisher INFORMATION for item j at ability theta measures how precisely
the item pins down theta:

    I_j(theta) = a_j^2 P_j(theta) (1 - P_j(theta))     for 2PL

Test information = sum of item informations:
    I(theta) = sum_j I_j(theta)

Standard error of theta_hat at that level:
    SE(theta) = 1 / sqrt(I(theta))

Applications
    - Adaptive testing (CAT): pick the next item to MAXIMIZE information
      at the examinee's current theta_hat -> shortens the test.
    - Test design: choose items to spread information evenly across the
      target theta range.
    - Report a TEST INFORMATION CURVE showing SE across theta.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)


def _sigmoid(x): return 1 / (1 + np.exp(-x))


def item_information_2pl(theta, a, b):
    """2PL item information at theta."""
    P = _sigmoid(a * (theta - b))
    return a ** 2 * P * (1 - P)


def test_information(theta_grid, a, b) -> dict:
    """Test information curve over a theta grid."""
    theta_grid = np.asarray(theta_grid, dtype=float)
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    I = np.zeros_like(theta_grid)
    per_item = np.zeros((len(a), len(theta_grid)))
    for j in range(len(a)):
        per_item[j] = item_information_2pl(theta_grid, a[j], b[j])
        I += per_item[j]
    SE = 1 / np.sqrt(np.maximum(I, 1e-8))
    return {"theta_grid": theta_grid, "test_info": I, "SE_theta": SE,
            "per_item_info": per_item,
            "method": "Test information function (2PL)"}


def optimal_next_item_cat(theta_hat: float, a, b, used: set) -> int:
    """Select the item (not yet used) with maximum information at theta_hat."""
    a = np.asarray(a); b = np.asarray(b)
    infos = np.array([item_information_2pl(theta_hat, a[j], b[j]) if j not in used else -1
                       for j in range(len(a))])
    return int(np.argmax(infos))


if __name__ == "__main__":
    # 15 items with varied a and b
    a = np.array([1.0, 1.2, 0.8, 1.5, 1.0, 1.3, 0.9, 1.1, 1.4, 1.0, 1.2, 0.7, 1.6, 1.1, 0.9])
    b = np.linspace(-2.5, 2.5, 15)

    grid = np.linspace(-3, 3, 13)
    r = test_information(grid, a, b)
    print("=== Test information function ===")
    print("  theta   I(theta)   SE(theta)")
    for t, I_, se in zip(grid, r["test_info"], r["SE_theta"]):
        print(f"  {t:5.2f}    {I_:6.3f}    {se:6.3f}")

    print("\n=== Adaptive next-item selection at theta_hat = 0.5 ===")
    idx = optimal_next_item_cat(0.5, a, b, used=set())
    print(f"  most informative item at theta=0.5: item {idx} (b = {b[idx]:.2f})")
    idx = optimal_next_item_cat(-1.5, a, b, used=set())
    print(f"  most informative item at theta=-1.5: item {idx} (b = {b[idx]:.2f})")

    print("\n--- library cross-check (R mirt::testinfo / plot(fit, type = 'info')) ---")
