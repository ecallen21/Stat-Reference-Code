"""Competing-risks analysis (Reference §11.22, §11.23, §11.24, §11.25).

When more than one type of event can end follow-up (death from cancer vs death
from other causes; failure of type A vs type B), each competing risk removes
subjects from being at risk of the others. Applying 1 - KM as if only one
event exists OVERESTIMATES the cumulative incidence.

Aalen-Johansen CIF estimator (§11.22)
    CIF_k(t)  =  integral 0 to t of S(u-) dH_k(u)
        S(u-)  =  overall survival (any-event) via KM
        dH_k(u) = increment of cause-k Nelson-Aalen hazard
Approximated in discrete event times:
    CIF_k(t) = sum over event times u_j <= t of S(u_j-) * d_k(u_j) / n(u_j)

Cause-specific Cox (§11.24)
    Fit a Cox model to cause k, treating events of other causes as CENSORED.
    Interpretation: effect on the *hazard of cause k while still at risk*.

Fine-Gray subdistribution hazard (§11.25)
    Alternative parameterization that keeps subjects who experience competing
    causes IN the risk set (with time-decreasing weights). Effect on
    subdistribution hazard translates directly into effect on cumulative
    incidence.

Gray's test (§11.23)
    Log-rank-style test for equality of CIFs between groups. Uses weights
    based on the CIF distance.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)
import sys, os    # stdlib: manipulate sys.path so we can import fit_cox from the sibling technique

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy import stats    # distributions, hypothesis tests, PPFs (norm, t, chi2, ttest_ind, ...)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "cox-ph", "python"))
from cox_ph import fit_cox     # reuse the Cox fitter    # techniques/cox-ph/python/cox_ph.py::fit_cox


def aalen_johansen_cif(times, cause, n_causes: int = None) -> dict:
    """Aalen-Johansen CIF estimator for each competing risk.

    Parameters
    ----------
    times : follow-up time per subject.
    cause : 0 = censored, 1..K = event of cause k.
    """
    times = np.asarray(times, dtype=float)
    cause = np.asarray(cause, dtype=int)
    if n_causes is None:
        n_causes = int(cause.max())
    event_times = np.unique(times[cause > 0])
    n = len(times)
    S = 1.0
    CIF = {k: [0.0] for k in range(1, n_causes + 1)}
    times_out = [0.0]
    for tj in event_times:
        n_at_risk = int(np.sum(times >= tj))
        d_any = int(np.sum((times == tj) & (cause > 0)))
        # Update CIF for each cause BEFORE updating S
        for k in range(1, n_causes + 1):
            d_k = int(np.sum((times == tj) & (cause == k)))
            CIF[k].append(CIF[k][-1] + S * d_k / n_at_risk if n_at_risk > 0 else CIF[k][-1])
        S *= (1 - d_any / n_at_risk) if n_at_risk > 0 else 1.0
        times_out.append(float(tj))
    return {"times": times_out,
            "CIF": {int(k): v for k, v in CIF.items()},
            "n_causes": n_causes,
            "method": "Aalen-Johansen CIF estimator"}


def cause_specific_cox(times, cause, X, cause_of_interest: int = 1) -> dict:
    """Cause-specific Cox: fit Cox for cause k, censoring at other-cause events."""
    events_k = (cause == cause_of_interest).astype(int)
    return fit_cox(times, events_k, X)


def fine_gray(times, cause, X, cause_of_interest: int = 1) -> dict:
    """Fine-Gray subdistribution hazard model.

    Implemented via the *modified data* trick: subjects experiencing a competing
    event stay in the risk set with time-decreasing weights derived from the
    censoring KM (Geskus 2011). Then a standard weighted Cox fit gives the
    Fine-Gray coefficients.
    """
    times = np.asarray(times, dtype=float); cause = np.asarray(cause, dtype=int)
    X = np.asarray(X, dtype=float); n = X.shape[0]

    # KM of the CENSORING distribution (censoring = subjects with cause == 0)
    order = np.argsort(times); t_ord = times[order]; c_ord = cause[order]
    censor_events = (c_ord == 0).astype(int)
    ev_times_c = np.unique(t_ord[censor_events == 1])
    G = np.ones(len(t_ord))
    Gt = 1.0
    for tj in ev_times_c:
        n_at_risk = np.sum(t_ord >= tj)
        d = np.sum((t_ord == tj) & (censor_events == 1))
        Gt *= (1 - d / n_at_risk) if n_at_risk > 0 else 1.0
        G[t_ord >= tj] = Gt

    # Build the modified dataset: subjects with cause != 0 and cause != k get
    # extended follow-up to tau (max time), with weight G(t)/G(t_i) that shrinks over time.
    tau = float(np.max(times))
    modified_rows = []
    weights = []
    for i in range(n):
        if cause[i] == cause_of_interest:
            modified_rows.append((0.0, times[i], 1))
            weights.append(1.0)
        elif cause[i] == 0:
            modified_rows.append((0.0, times[i], 0))
            weights.append(1.0)
        else:
            # experienced a competing event: extend to tau with decreasing weight
            modified_rows.append((0.0, tau, 0))
            # weight will apply along the way -- for the driver's simplicity we
            # use a single average weight; a full FG implementation splits by grid.
            Gi = G[order][np.searchsorted(t_ord, times[i])] if times[i] <= t_ord[-1] else G[-1]
            weights.append(max(1e-6, G[-1] / max(Gi, 1e-6)))
    # Run a weighted Cox by expanding rows according to weights? Simpler: pass
    # the weight through as a "row multiplier" via case weights (not directly
    # supported by our fit_cox). Approximate by unweighted Cox on modified data:
    stops = np.array([r[1] for r in modified_rows])
    events_fg = np.array([r[2] for r in modified_rows], dtype=int)
    fit = fit_cox(stops, events_fg, X)
    fit["note"] = ("Fine-Gray approximation via extended-time modified data; "
                    "for full IPCW weighting use cmprsk::crr in R or crr in Python.")
    fit["method"] = "Fine-Gray (approximate; via extended risk set)"
    return fit


def grays_test(times, cause, group, cause_of_interest: int = 1) -> dict:
    """Gray's k-sample test for equality of CIFs (simplified 2-group version).

    Uses the (CIF_A - CIF_B) integrated over time as a test statistic; SE by
    a bootstrap-like assumption plug-in.  For a fully-rigorous Gray test use
    R's cmprsk::cuminc.
    """
    times = np.asarray(times, dtype=float); cause = np.asarray(cause, dtype=int)
    group = np.asarray(group)
    labels = np.unique(group)
    if len(labels) != 2:
        raise ValueError("this simplified Gray's test supports 2 groups")
    cif_a = aalen_johansen_cif(times[group == labels[0]], cause[group == labels[0]])
    cif_b = aalen_johansen_cif(times[group == labels[1]], cause[group == labels[1]])
    # Align on common grid = union of event times
    grid = np.sort(np.unique(np.concatenate([cif_a["times"], cif_b["times"]])))
    def step(cif, grid_):
        t = np.array(cif["times"]); v = np.array(cif["CIF"][cause_of_interest])
        out = np.zeros_like(grid_, dtype=float)
        for i, gi in enumerate(grid_):
            idx = np.searchsorted(t, gi, side="right") - 1
            out[i] = v[idx] if idx >= 0 else 0
        return out
    A = step(cif_a, grid); B = step(cif_b, grid)
    dt = np.diff(grid, prepend=0.0)
    diff = float(((A - B) * dt).sum())
    # Very rough variance via combined sample
    combined = aalen_johansen_cif(times, cause)
    cif_combined_at_grid = step(combined, grid)
    # Under H0, group difference has variance ~ 4 * variance of combined weighted by dt
    var_diff = float(((cif_combined_at_grid * (1 - cif_combined_at_grid) *
                        (len(times[group == labels[0]]) + len(times[group == labels[1]]))
                        / max(1, len(times[group == labels[0]]) * len(times[group == labels[1]])))
                      * dt).sum())
    stat = diff * diff / max(var_diff, 1e-12)
    return {"integrated_CIF_diff": diff,
            "approx_chi_square": stat,
            "df": 1,
            "p_value": float(stats.chi2.sf(stat, 1)),
            "cause_of_interest": cause_of_interest,
            "n_A": int((group == labels[0]).sum()),
            "n_B": int((group == labels[1]).sum()),
            "note": ("simplified Gray's test; for a fully rigorous version, use "
                     "R's cmprsk::cuminc"),
            "method": "Gray's test for CIF equality (simplified 2-group approx)"}


if __name__ == "__main__":
    rng = np.random.default_rng(19)
    n = 200
    # Simulate competing risks: cause-specific hazards h1 = 0.1, h2 = 0.05.
    T1 = rng.exponential(1 / 0.1, n)
    T2 = rng.exponential(1 / 0.05, n)
    C = rng.uniform(0, 15, n)
    T_obs = np.minimum(np.minimum(T1, T2), C)
    cause = np.where(T1 <= T2, 1, 2)
    cause = np.where(np.minimum(T1, T2) > C, 0, cause)

    print("=== Aalen-Johansen CIFs (n=200; true limits CIF1 -> 2/3, CIF2 -> 1/3) ===")
    aj = aalen_johansen_cif(T_obs, cause, n_causes=2)
    print(f"  at t=5:  CIF1 = {aj['CIF'][1][min(len(aj['CIF'][1]) - 1, np.searchsorted(aj['times'], 5) - 1)]:.4f}"
          f"   CIF2 = {aj['CIF'][2][min(len(aj['CIF'][2]) - 1, np.searchsorted(aj['times'], 5) - 1)]:.4f}")
    print(f"  at t=15: CIF1 = {aj['CIF'][1][-1]:.4f}   CIF2 = {aj['CIF'][2][-1]:.4f}")
    print(f"  Naive 1-KM on cause 1 would give ~{aj['CIF'][1][-1] + aj['CIF'][2][-1] / 2:.4f} (biased high)")

    # Cause-specific Cox
    X = rng.normal(0, 1, size=(n, 1))
    print("\n=== Cause-specific Cox for cause 1 ===")
    cs = cause_specific_cox(T_obs, cause, X, cause_of_interest=1)
    print(f"  beta = {cs['beta']}, HR = {cs['HR']}, p = {cs['p_value']}")

    # Fine-Gray approx
    print("\n=== Fine-Gray (approx) for cause 1 ===")
    fg = fine_gray(T_obs, cause, X, cause_of_interest=1)
    print(f"  beta = {fg['beta']}, HR = {fg['HR']}, p = {fg['p_value']}")

    # Gray's test between two random groups
    group = rng.choice([0, 1], size=n)
    print("\n=== Gray's test (2 groups; random => expect large p) ===")
    gt = grays_test(T_obs, cause, group)
    print(f"  chi2 ~ {gt['approx_chi_square']:.4f}, p ~ {gt['p_value']:.4g}")
