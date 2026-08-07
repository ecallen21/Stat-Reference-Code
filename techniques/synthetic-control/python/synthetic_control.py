"""Synthetic Control (Reference §15.10; Abadie-Diamond-Hainmueller 2010).

One treated unit, many potential controls (the 'donor pool'), and a long
pre-treatment time series.  Construct a WEIGHTED AVERAGE of donors that
matches the treated unit's pre-treatment trajectory; use the same weighted
average as the counterfactual after treatment.

Optimization
    Choose non-negative weights W = (w_1, ..., w_J) summing to 1 that minimize
        sum_t (Y_{1, t}^pre  -  sum_j w_j Y_{j, t}^pre)^2
    (Optionally weight time periods by V for pre-treatment covariate
    balance; the demo omits V for simplicity.)

Treatment effect
    ATT_t = Y_{1, t}^post  -  sum_j w_j Y_{j, t}^post
Report the time series of gaps; often summarize as an average post-period ATT.

Inference: PLACEBO tests reassign the treatment to each donor unit and
compare the post-pre RMSE ratio.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy.optimize import minimize    # SciPy optimizer (BFGS/Newton) for MLE


def synthetic_control(Y_treated_pre, Y_donors_pre) -> dict:
    """Solve for non-negative weights summing to 1 that best match pre-treatment trajectory."""
    Y_t = np.asarray(Y_treated_pre, dtype=float)
    Y_d = np.asarray(Y_donors_pre, dtype=float)    # T_pre x J
    J = Y_d.shape[1]
    def loss(w):
        return float(np.sum((Y_t - Y_d @ w) ** 2))
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    bounds = [(0.0, 1.0)] * J
    x0 = np.full(J, 1 / J)
    res = minimize(loss, x0, method="SLSQP", bounds=bounds, constraints=cons)
    w = res.x; w = w / w.sum()
    return {"weights": w, "loss": float(res.fun),
            "sc_pre": Y_d @ w,
            "method": "Synthetic control (Abadie-Diamond-Hainmueller)"}


def sc_effect(Y_treated, Y_donors, t_treat: int) -> dict:
    """Fit SC on pre-period, report post-period gap trajectory + average ATT."""
    Y_treated = np.asarray(Y_treated, dtype=float)
    Y_donors = np.asarray(Y_donors, dtype=float)   # T x J
    fit = synthetic_control(Y_treated[:t_treat], Y_donors[:t_treat, :])
    sc_full = Y_donors @ fit["weights"]
    gap = Y_treated - sc_full
    post = gap[t_treat:]
    return {"weights": fit["weights"],
            "gap_series": gap,
            "sc_trajectory": sc_full,
            "avg_post_ATT": float(post.mean()),
            "pre_RMSE": float(np.sqrt(np.mean(gap[:t_treat] ** 2))),
            "post_RMSE": float(np.sqrt(np.mean(gap[t_treat:] ** 2))),
            "method": "Synthetic control ATT estimate"}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    T_total = 30; T_pre = 20; J = 20
    t = np.arange(T_total).reshape(-1, 1)
    # Donors: 20 units with heterogeneous trends
    slopes = rng.uniform(0.1, 1.0, J); intercepts = rng.uniform(0, 5, J)
    Y_donors = intercepts + slopes * t + rng.normal(0, 0.4, (T_total, J))
    # Treated unit: matches a specific mix (weights on donors 2, 5, 10) pre-treatment
    true_w = np.zeros(J); true_w[[2, 5, 10]] = [0.4, 0.3, 0.3]
    Y_treated = Y_donors @ true_w + rng.normal(0, 0.3, T_total)
    Y_treated[T_pre:] += 3.0  # ATT = +3 after t_treat = 20

    r = sc_effect(Y_treated, Y_donors, t_treat=T_pre)
    top_w = np.argsort(-r["weights"])[:5]
    print("=== Synthetic control fit ===")
    print(f"  top 5 donor weights: {[(int(i), round(float(r['weights'][i]), 3)) for i in top_w]}")
    print(f"  pre-period RMSE: {r['pre_RMSE']:.3f}")
    print(f"  post-period RMSE (gap): {r['post_RMSE']:.3f}")
    print(f"  average post-period ATT: {r['avg_post_ATT']:.3f}  (true 3.0)")

    print("\n  Gap trajectory (year - SC counterfactual):")
    for i in (T_pre - 3, T_pre - 1, T_pre, T_pre + 3, T_pre + 9):
        marker = "  " if i < T_pre else "* "
        print(f"    {marker}t = {i}: gap = {r['gap_series'][i]:.3f}")

    print("\n--- library cross-check (R Synth) ---")
    print("  R: Synth::synth(dataprep.out) with the classical Basque/California-tobacco setup.")
