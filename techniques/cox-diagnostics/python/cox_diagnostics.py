"""Cox model residuals + PH-assumption test (Reference §11.33; also covers §11.53).

Four common Cox-model residual types + the Grambsch-Therneau PH test:

1. Schoenfeld residuals (one per event, one per covariate)
       r_{Sk,l}(t_j)  =  X_{jl} - E[X_l | R(t_j), event]
   Diagnostic for PROPORTIONAL HAZARDS. Trended-with-time residuals imply
   the log-HR is drifting -> PH violated.

2. Scaled Schoenfeld (Grambsch-Therneau 1994)
       r*_{Sk,l}  =  d * cov(beta) * r_S + beta
   Their correlation with a function g(t) (typically log-t) is the basis for a
   chi-square PH test.

3. Martingale residuals (one per subject)
       r_M_i = event_i - integral over 0 to t_i of exp(X_i beta) dH0(u)
   Diagnostic for FUNCTIONAL FORM: plot r_M vs. a continuous covariate; a
   non-linear trend suggests you should transform or spline that covariate.

4. Cox-Snell residuals
       r_CS_i  =  integral 0 to t_i of exp(X_i beta) dH0(u)  =  H_hat(t_i | X_i)
   If the model is well-specified, r_CS should behave like a censored sample
   from Exp(1) -- their KM plotted against exp(-r) should be near y = x.

5. Deviance residuals
       r_D_i = sign(r_M_i) * sqrt(-2 (r_M_i + event_i * log(event_i - r_M_i)))
   Approximately symmetric around 0; useful for outlier detection.

We reuse the Cox partial-likelihood machinery from techniques/cox-ph.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)
import sys, os    # stdlib: manipulate sys.path so we can import from the sibling technique

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy import stats    # distributions, hypothesis tests, PPFs (norm, t, chi2, ttest_ind, ...)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "cox-ph", "python"))
from cox_ph import fit_cox    # reuse the fitter    # techniques/cox-ph/python/cox_ph.py::fit_cox


def _breslow_baseline_cumhazard(times, events, X, beta):
    """H0(t) at each event time via the Breslow estimator (given beta)."""
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    exp_eta = np.exp(np.clip(X @ beta, -500, 500))
    event_times = np.unique(times[events == 1])
    H0 = np.zeros(len(event_times))
    cum = 0.0
    for j, tj in enumerate(event_times):
        R = times >= tj
        d_j = int(np.sum((times == tj) & (events == 1)))
        cum += d_j / max(exp_eta[R].sum(), 1e-300)
        H0[j] = cum
    return event_times, H0


def schoenfeld_residuals(times, events, X, beta) -> dict:
    """One Schoenfeld residual per event per covariate."""
    X = np.asarray(X, dtype=float)
    exp_eta = np.exp(np.clip(X @ beta, -500, 500))
    event_times = np.unique(times[events == 1])
    resids = []
    tj_list = []
    for tj in event_times:
        R = times >= tj
        e_R = exp_eta[R]; X_R = X[R]
        Ebar = (e_R[:, None] * X_R).sum(axis=0) / e_R.sum()
        d_indices = np.where((times == tj) & (events == 1))[0]
        for i in d_indices:
            resids.append(X[i] - Ebar)
            tj_list.append(tj)
    return {"event_times": tj_list,
            "schoenfeld": np.array(resids).tolist(),
            "method": "Schoenfeld residuals (per event, per covariate)"}


def grambsch_therneau_ph_test(times, events, X, beta, cov_beta) -> dict:
    """Grambsch-Therneau PH-assumption test using scaled Schoenfeld residuals
    correlated with a monotone function of time (default: identity of rank(t))."""
    sr = schoenfeld_residuals(times, events, X, beta)
    r = np.array(sr["schoenfeld"])         # d x p
    d, p = r.shape
    if d == 0:
        return {"per_covariate_p": [], "global_chi2": float("nan"), "df": p,
                 "global_p": float("nan"),
                 "method": "Grambsch-Therneau (no events)"}
    # scaled Schoenfeld: r_scaled = d * cov * r'
    r_scaled = (d * cov_beta @ r.T).T + beta          # d x p (Wald-style scaling)
    # rank(time) as g(t) -- robust monotone transform
    g = stats.rankdata(np.array(sr["event_times"])) - (d + 1) / 2
    per_cov_stats = []
    per_cov_p = []
    for k in range(p):
        num = float((g * (r_scaled[:, k] - r_scaled[:, k].mean())).sum())
        denom = float((g ** 2).sum() * cov_beta[k, k] * d)
        chi2_k = num * num / denom if denom > 0 else float("nan")
        per_cov_stats.append(chi2_k)
        per_cov_p.append(float(stats.chi2.sf(chi2_k, 1)) if math.isfinite(chi2_k) else float("nan"))
    global_chi2 = float(sum(s for s in per_cov_stats if math.isfinite(s)))
    return {"per_covariate_chi_square": per_cov_stats,
             "per_covariate_p_value": per_cov_p,
             "global_chi_square": global_chi2,
             "global_df": p,
             "global_p_value": float(stats.chi2.sf(global_chi2, p)),
             "method": "Grambsch-Therneau (1994) PH test on scaled Schoenfeld residuals"}


def martingale_residuals(times, events, X, beta) -> dict:
    """r_M_i = event_i - H0(t_i) * exp(X_i beta)."""
    X = np.asarray(X, dtype=float)
    ev_times, H0 = _breslow_baseline_cumhazard(times, events, X, beta)
    # step-function H0(t): value = latest H0 at event_time <= t (0 before first)
    idx = np.searchsorted(ev_times, times, side="right") - 1
    H0_at_t = np.where(idx >= 0, H0[np.clip(idx, 0, len(H0) - 1)], 0.0)
    exp_eta = np.exp(np.clip(X @ beta, -500, 500))
    rM = events - H0_at_t * exp_eta
    return {"martingale": rM.tolist(),
            "method": "martingale residuals"}


def cox_snell_residuals(times, events, X, beta) -> dict:
    """r_CS_i = H_hat(t_i | X_i) = H0(t_i) * exp(X_i beta). Should behave like Exp(1) if
    the model is well-specified (check via KM of r_CS vs. exp(-r))."""
    X = np.asarray(X, dtype=float)
    ev_times, H0 = _breslow_baseline_cumhazard(times, events, X, beta)
    idx = np.searchsorted(ev_times, times, side="right") - 1
    H0_at_t = np.where(idx >= 0, H0[np.clip(idx, 0, len(H0) - 1)], 0.0)
    exp_eta = np.exp(np.clip(X @ beta, -500, 500))
    rCS = H0_at_t * exp_eta
    return {"cox_snell": rCS.tolist(),
            "mean_if_wellspecified": 1.0,
            "method": "Cox-Snell residuals (should be Exp(1) if model is OK)"}


def deviance_residuals(martingale, events) -> list:
    rM = np.asarray(martingale); e = np.asarray(events)
    with np.errstate(invalid="ignore"):
        rD = np.sign(rM) * np.sqrt(np.clip(-2 * (rM + np.where(e > 0, e * np.log(np.maximum(e - rM, 1e-12)), 0.0)), 0.0, None))
    return rD.tolist()


def library_versions(times, events, X):
    try:
        from lifelines import CoxPHFitter
        import pandas as pd
        df = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
        df["T"] = times; df["E"] = events
        cph = CoxPHFitter().fit(df, duration_col="T", event_col="E")
        ph = cph.check_assumptions(df, p_value_threshold=0.05, show_plots=False)
        return {"lifelines PH check": "see summary above (returns a DataFrame)"}
    except Exception as ex:
        return {"lifelines (optional)": f"not available: {ex}"}


if __name__ == "__main__":
    rng = np.random.default_rng(13)
    n = 300
    X = rng.normal(0, 1, size=(n, 2))
    lin = 0.6 * X[:, 0] - 0.3 * X[:, 1]
    U = rng.uniform(0, 1, n)
    T_event = -np.log(U) / (0.1 * np.exp(lin))
    C_censor = rng.uniform(0, 20, n)
    times = np.minimum(T_event, C_censor)
    events = (T_event <= C_censor).astype(int)
    fit = fit_cox(times, events, X, ties="efron")
    beta = np.array(fit["beta"])
    p = X.shape[1]
    cov_beta = np.array([[s * s if i == j else 0.0
                          for j, s in enumerate(fit["SE"])]
                         for i, _ in enumerate(fit["SE"])])   # diagonal SE^2 (crude)

    print(f"=== Cox fit: beta = {[f'{b:+.4f}' for b in beta]} ===")

    print("\n=== Grambsch-Therneau PH test (should NOT reject; hazards are PH here) ===")
    gt = grambsch_therneau_ph_test(times, events, X, beta, cov_beta)
    for k, (c2, pv) in enumerate(zip(gt["per_covariate_chi_square"], gt["per_covariate_p_value"])):
        print(f"  x{k}: chi2 = {c2:.4f}, p = {pv:.4g}")
    print(f"  GLOBAL: chi2 = {gt['global_chi_square']:.4f} on df = {gt['global_df']}, p = {gt['global_p_value']:.4g}")

    print("\n=== Residuals (first 5 per type) ===")
    mr = martingale_residuals(times, events, X, beta)
    csr = cox_snell_residuals(times, events, X, beta)
    dr = deviance_residuals(mr["martingale"], events)
    print(f"  martingale : {[f'{r:+.3f}' for r in mr['martingale'][:5]]}")
    print(f"  cox-snell  : {[f'{r:.3f}' for r in csr['cox_snell'][:5]]}")
    print(f"  deviance   : {[f'{r:+.3f}' for r in dr[:5]]}")
    print(f"  (sum(martingale) should be ~ 0):  {sum(mr['martingale']):+.4f}")
