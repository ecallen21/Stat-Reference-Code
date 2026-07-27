"""Parametric AFT survival models (Reference §11.10-§11.15; also covers §11.44, §11.58).

Accelerated Failure Time parameterization:
    log T  =  X * beta  +  sigma * W
where W has a family-specific distribution:
    exp        -> W = extreme-value (Gumbel);  sigma = 1
    Weibull    -> W = extreme-value (Gumbel);  sigma free
    log-normal -> W = N(0, 1)
    log-logist -> W = logistic
    gen. gamma -> W has one extra shape parameter

MLE via BFGS on the observed-data log-likelihood with censoring:
    ll_i  =  log f(t_i | X_i, params)     if event
             log S(t_i | X_i, params)     if censored

Piecewise-exponential model (§11.44): partition [0, tau_max] into intervals;
constant hazard within each interval; equivalent to a Poisson-GLM
formulation on person-time.

Weibull reliability (§11.58): the Weibull is the classical reliability model.
The 'reliability plot' is log(-log S) vs. log t and should be linear with slope
equal to the Weibull SHAPE parameter.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy import optimize, stats    # optimize: BFGS;  stats: distributions/tests


# --- Family-specific f, S, and log(f) helpers, all on ORIGINAL scale ------

def _neg_ll_aft(params, X, times, events, family: str):
    """Negative log-likelihood for the specified AFT family."""
    n, p = X.shape
    beta = params[:p]; log_sigma = params[p]
    sigma = math.exp(log_sigma)
    linpred = X @ beta                         # this is log-scale location
    z = (np.log(np.clip(times, 1e-300, None)) - linpred) / sigma

    if family == "weibull":
        # T ~ Weibull with shape 1/sigma, scale exp(linpred). Density on log-t is Gumbel(0,1).
        # log f_T(t) = log( f_W(z) / (sigma * t) ) ,  S(t) = exp(-exp(z))
        log_f = z - np.exp(z) - np.log(sigma * np.clip(times, 1e-300, None))
        log_S = -np.exp(z)
    elif family == "exponential":
        # Weibull with sigma = 1
        sigma = 1.0
        z = (np.log(np.clip(times, 1e-300, None)) - linpred)
        log_f = z - np.exp(z) - np.log(np.clip(times, 1e-300, None))
        log_S = -np.exp(z)
    elif family == "lognormal":
        # log T ~ N(linpred, sigma^2). f_T(t) = phi(z) / (sigma t).  S(t) = 1 - Phi(z).
        log_f = -0.5 * z * z - 0.5 * math.log(2 * math.pi) - np.log(sigma * np.clip(times, 1e-300, None))
        # log(1 - Phi(z)) numerically stable via norm.logsf
        log_S = stats.norm.logsf(z)
    elif family == "loglogistic":
        # log T ~ Logistic(linpred, sigma).  f_T(t) = f_L(z) / (sigma t).  S(t) = 1/(1 + exp(z)).
        # log f_L(z) = -z - 2 log(1 + exp(-z))
        log_f = -z - 2 * np.log1p(np.exp(-z)) - np.log(sigma * np.clip(times, 1e-300, None))
        log_S = -np.log1p(np.exp(z))
    else:
        raise ValueError("family must be exponential / weibull / lognormal / loglogistic")

    ll = np.where(events == 1, log_f, log_S).sum()
    return -float(ll)


def fit_aft(times, events, X=None, family: str = "weibull", intercept: bool = True) -> dict:
    """Fit an AFT model by MLE.

    Parameters
    ----------
    times, events : follow-up times and event indicator (0/1).
    X : n x p design matrix (WITHOUT intercept unless ``intercept=False``); if
        None, model has intercept only.
    family : 'exponential' / 'weibull' / 'lognormal' / 'loglogistic'.
    """
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=int)
    if X is None:
        Xd = np.ones((len(times), 1))
    else:
        X = np.asarray(X, dtype=float)
        Xd = np.column_stack([np.ones(len(times)), X]) if intercept else X
    p = Xd.shape[1]
    # exponential has fixed sigma = 1; keep log_sigma slot but constrain via family
    theta0 = np.zeros(p + 1)
    theta0[0] = float(np.log(np.median(times)))  # intercept starting value
    res = optimize.minimize(_neg_ll_aft, theta0, args=(Xd, times, events, family),
                             method="BFGS", options={"gtol": 1e-7})
    beta = res.x[:p]; log_sigma = res.x[p]
    sigma = math.exp(log_sigma) if family != "exponential" else 1.0
    cov = res.hess_inv
    se_all = np.sqrt(np.clip(np.diag(cov), 0, None))
    return {"family": family,
            "beta": beta.tolist(),
            "sigma": sigma,
            "SE_beta": se_all[:p].tolist(),
            "SE_log_sigma": float(se_all[p]) if family != "exponential" else 0.0,
            "log_lik": float(-res.fun),
            "n": int(len(times)), "n_events": int((events == 1).sum()),
            "n_params": p + (0 if family == "exponential" else 1),
            "AIC": 2 * (p + (0 if family == "exponential" else 1)) + 2 * res.fun,
            "method": f"AFT MLE via BFGS ({family})"}


def piecewise_exponential(times, events, breakpoints) -> dict:
    """Piecewise-constant hazard model (§11.44).

    Fit K hazards, one per interval defined by ``breakpoints`` (list of cut times).
    Implemented as a Poisson GLM on person-time: for each subject split their
    follow-up into interval-specific rows and fit rate_j = events_j / persontime_j.
    """
    times = np.asarray(times, dtype=float); events = np.asarray(events, dtype=int)
    bps = np.array(list(breakpoints), dtype=float)
    bps = np.concatenate([[0.0], bps, [np.inf]])
    hazards = []
    for j in range(len(bps) - 1):
        lo, hi = bps[j], bps[j + 1]
        pt = np.clip(times, lo, hi) - lo
        pt = np.where(pt > 0, pt, 0)          # zero contribution before subject exists in this bin
        ev = ((times >= lo) & (times < hi) & (events == 1)).astype(int)
        total_pt = float(pt.sum())
        total_ev = int(ev.sum())
        h = total_ev / total_pt if total_pt > 0 else float("nan")
        hazards.append({"interval": (float(lo), float(hi)),
                        "events": total_ev,
                        "person_time": total_pt,
                        "hazard": h})
    return {"intervals": hazards,
            "method": "piecewise-exponential MLE per bin"}


def weibull_reliability_plot_data(times, events) -> dict:
    """§11.58: (log t, log(-log S_hat(t))) for a Weibull-goodness reliability plot.

    On log-log paper, a well-fit Weibull is a straight line with slope = shape.
    """
    ord_ = np.argsort(times); t = times[ord_]; e = events[ord_]
    S = 1.0; xs = []; ys = []
    n = len(t)
    for i, ti in enumerate(t):
        if e[i] == 1:
            n_j = n - i
            S *= (1 - 1 / n_j)
            if 0 < S < 1:
                xs.append(math.log(ti))
                ys.append(math.log(-math.log(S)))
    # Simple slope estimate (should ~ Weibull shape)
    if len(xs) > 2:
        slope, intercept, *_ = stats.linregress(xs, ys)
    else:
        slope = intercept = float("nan")
    return {"log_t": xs, "log_minus_log_S": ys,
            "estimated_shape_slope": float(slope),
            "intercept": float(intercept),
            "method": "Weibull reliability-plot data (log-log)"}


def library_versions(times, events):
    try:
        from lifelines import WeibullFitter, LogNormalFitter, LogLogisticFitter, ExponentialFitter
        out = {}
        for name, cls in [("Weibull", WeibullFitter), ("Exponential", ExponentialFitter),
                           ("LogNormal", LogNormalFitter), ("LogLogistic", LogLogisticFitter)]:
            f = cls().fit(times, events)
            out[f"lifelines {name} log-lik"] = float(f.log_likelihood_)
        return out
    except Exception as ex:
        return {"lifelines (optional)": f"not available: {ex}"}


if __name__ == "__main__":
    rng = np.random.default_rng(17)
    n = 300
    # Weibull DGP: shape = 1.5, scale = 10
    shape_true = 1.5; scale_true = 10.0
    T_event = scale_true * (-np.log(rng.uniform(0, 1, n))) ** (1 / shape_true)
    C_censor = rng.uniform(0, 20, n)
    times = np.minimum(T_event, C_censor)
    events = (T_event <= C_censor).astype(int)

    print("=== AFT models (intercept only) ===")
    for fam in ("exponential", "weibull", "lognormal", "loglogistic"):
        r = fit_aft(times, events, X=None, family=fam)
        print(f"  {fam:12s}: ll = {r['log_lik']:8.3f}   AIC = {r['AIC']:8.3f}   sigma = {r['sigma']:.4f}")

    print(f"\n=== Weibull-specific: true (shape={shape_true}, scale={scale_true}) ===")
    wr = fit_aft(times, events, X=None, family="weibull")
    beta0 = wr["beta"][0]; sig = wr["sigma"]
    est_scale = math.exp(beta0)
    est_shape = 1 / sig
    print(f"  estimated scale (exp intercept) = {est_scale:.4f}")
    print(f"  estimated shape (1 / sigma)     = {est_shape:.4f}")

    print("\n=== Weibull reliability-plot slope (should ~ shape) ===")
    rp = weibull_reliability_plot_data(times, events)
    print(f"  slope = {rp['estimated_shape_slope']:.4f}   (true shape = {shape_true})")

    print("\n=== Piecewise-exponential (breakpoints at 2, 5, 10) ===")
    pe = piecewise_exponential(times, events, [2, 5, 10])
    for row in pe["intervals"]:
        print(f"  {row}")

    print("\n--- library (lifelines) ---")
    for k, v in library_versions(times, events).items():
        print(f"  {k}: {v}")
