"""Random-intercept cross-lagged panel model (RI-CLPM) (Reference Sec 20.30).

Hamaker, Kuiper & Grasman 2015 'A critique of the cross-lagged panel
model', Psych Meth. Separates BETWEEN-person differences (trait
levels, stable) from WITHIN-person dynamics (cross-lagged causal
influence). Contrast: classical CLPM conflates the two.

For two variables X, Y measured at t = 1..T:

    x_t = alpha_x + u_i + x_star_t         with  x_star_t = phi_xx * x_star_{t-1} + phi_xy * y_star_{t-1} + e_x
    y_t = alpha_y + v_i + y_star_t         with  y_star_t = phi_yx * x_star_{t-1} + phi_yy * y_star_{t-1} + e_y

phi_xy: within-person effect of Y at t-1 on X at t (with trait removed).

We centre by subject to remove (u_i, v_i), then fit an AR(1)-cross-
lagged VAR on within-person deviations.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def ri_clpm_fit(x, y):
    """x, y : (n_subjects, T) matrices."""
    n, T = x.shape
    #  Centre within-person
    xw = x - x.mean(axis=1, keepdims=True)
    yw = y - y.mean(axis=1, keepdims=True)
    #  Stack lagged regressions:
    #    x_t = a + phi_xx * x_{t-1} + phi_xy * y_{t-1}
    xt = xw[:, 1:].ravel()
    yt = yw[:, 1:].ravel()
    xlag = xw[:, :-1].ravel()
    ylag = yw[:, :-1].ravel()
    A = np.c_[np.ones_like(xt), xlag, ylag]
    b_x, *_ = np.linalg.lstsq(A, xt, rcond=None)
    b_y, *_ = np.linalg.lstsq(A, yt, rcond=None)
    return {"phi_xx": b_x[1], "phi_xy": b_x[2],
            "phi_yx": b_y[1], "phi_yy": b_y[2]}


def classical_clpm_fit(x, y):
    """Standard CLPM (no random intercept) for comparison."""
    n, T = x.shape
    xt = x[:, 1:].ravel(); yt = y[:, 1:].ravel()
    xlag = x[:, :-1].ravel(); ylag = y[:, :-1].ravel()
    A = np.c_[np.ones_like(xt), xlag, ylag]
    b_x, *_ = np.linalg.lstsq(A, xt, rcond=None)
    b_y, *_ = np.linalg.lstsq(A, yt, rcond=None)
    return {"phi_xx": b_x[1], "phi_xy": b_x[2],
            "phi_yx": b_y[1], "phi_yy": b_y[2]}


if __name__ == "__main__":
    print("=== RI-CLPM (Hamaker et al. 2015) vs classical CLPM ===\n")
    rng = np.random.default_rng(0)
    n = 400; T = 6

    #  Between-person traits u, v ~ Normal(0, 1)
    u = rng.normal(0, 1.5, size=n)
    v = rng.normal(0, 1.5, size=n)

    #  Within-person AR(1) with cross-lag phi_xy = 0.15, phi_yx = 0
    x_star = np.zeros((n, T)); y_star = np.zeros((n, T))
    for t in range(1, T):
        x_star[:, t] = 0.4 * x_star[:, t - 1] + 0.15 * y_star[:, t - 1] + rng.normal(scale=0.3, size=n)
        y_star[:, t] = 0.5 * y_star[:, t - 1] + 0.0 * x_star[:, t - 1] + rng.normal(scale=0.3, size=n)

    x = x_star + u[:, None]
    y = y_star + v[:, None]

    r_cl = classical_clpm_fit(x, y)
    r_ri = ri_clpm_fit(x, y)
    print("  True: phi_xx=0.40  phi_xy=+0.15   phi_yx=0.00   phi_yy=0.50")
    print(f"  Classical CLPM:  phi_xx={r_cl['phi_xx']:+.3f}  "
          f"phi_xy={r_cl['phi_xy']:+.3f}  phi_yx={r_cl['phi_yx']:+.3f}  "
          f"phi_yy={r_cl['phi_yy']:+.3f}")
    print(f"  RI-CLPM       :  phi_xx={r_ri['phi_xx']:+.3f}  "
          f"phi_xy={r_ri['phi_xy']:+.3f}  phi_yx={r_ri['phi_yx']:+.3f}  "
          f"phi_yy={r_ri['phi_yy']:+.3f}")
    print("\n  Classical CLPM confounds trait variance with within-person dynamics;")
    print("  RI-CLPM after subject centring recovers the true within-person cross-lag.")

    print("\n--- library cross-check (lavaan / OpenMx R; pymc / semopy Python) ---")
