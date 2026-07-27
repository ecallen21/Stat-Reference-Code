"""Weighted log-rank family + stratified log-rank (Reference §11.4, §11.5, §11.6, §11.7).

For two groups A and B with right-censored data, compare the survival
distributions via a weighted rank statistic:

    U = sum over event times t_j of  w_j (d_{Aj} - E[d_{Aj}])
        where E[d_{Aj}] = n_{Aj} * d_j / n_j       under H0
        and Var contribution: w_j^2 * (n_{Aj} n_{Bj} d_j (n_j - d_j)) / (n_j^2 (n_j - 1))

Chi-square: (U^2 / Var) ~ chi2_1 under H0.

Weight choices (§11.47, §11.62):
    w_j = 1                       : standard LOG-RANK / Mantel-Cox (§11.4).
                                     Most powerful when hazards are proportional.
    w_j = n_j                     : Gehan-Breslow-Wilcoxon (§11.5).
                                     More weight on EARLY events.
    w_j = S_pooled(t_j-)          : Peto-Peto (§11.5).
                                     Sensitive to early differences, robust.
    w_j = (S_pooled)^rho *
          (1 - S_pooled)^gamma    : Fleming-Harrington G(rho, gamma) (§11.6).
                                     rho=1, gamma=0 -> late-emphasis.
                                     rho=0, gamma=1 -> early-emphasis.
                                     rho=0, gamma=0 -> log-rank.
    w_j = sqrt(n_j)               : Tarone-Ware (§11.62).

Stratified log-rank (§11.7): sum U and Var across strata, then chi-square.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)
from typing import Callable, Sequence    # stdlib: type hints

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy import stats    # distributions, hypothesis tests, PPFs (norm, t, chi2, ttest_ind, ...)


def _pooled_km_at(times, events, event_times):
    """Left-limit S_pooled(t_j-) for each event time (used by Peto-Peto)."""
    order = np.argsort(times); t = times[order]; e = events[order]
    S_prev = 1.0
    out = np.empty(len(event_times))
    for i, tj in enumerate(event_times):
        # S(t_j-) is S up to (but not including) events at t_j
        S = 1.0
        for tk in event_times:
            if tk >= tj: break
            n_k = np.sum(t >= tk); d_k = np.sum((t == tk) & (e == 1))
            S *= (1 - d_k / n_k) if n_k > 0 else 1
        out[i] = S
    return out


def log_rank_test(times, events, group, weight_scheme: str = "logrank",
                   rho: float = 0.0, gamma: float = 0.0) -> dict:
    """Weighted 2-sample log-rank test.

    ``weight_scheme``: 'logrank', 'wilcoxon' (Gehan-Breslow), 'peto', 'fh' (Fleming-
        Harrington with rho, gamma), 'tarone-ware'.
    """
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=int)
    group = np.asarray(group)
    labels = np.unique(group)
    if len(labels) != 2:
        raise ValueError("group must have exactly 2 distinct labels")
    a_label = labels[0]
    event_times = np.unique(times[events == 1])
    if weight_scheme == "peto":
        S_prev = _pooled_km_at(times, events, event_times)
    U = 0.0; V = 0.0
    for j, tj in enumerate(event_times):
        n_j = int(np.sum(times >= tj))
        d_j = int(np.sum((times == tj) & (events == 1)))
        n_Aj = int(np.sum((times >= tj) & (group == a_label)))
        d_Aj = int(np.sum((times == tj) & (events == 1) & (group == a_label)))
        if n_j <= 1 or d_j == 0: continue
        E_Aj = n_Aj * d_j / n_j
        var_j = (n_Aj * (n_j - n_Aj) * d_j * (n_j - d_j)) / (n_j * n_j * (n_j - 1))
        if weight_scheme == "logrank":       w_j = 1.0
        elif weight_scheme == "wilcoxon":    w_j = float(n_j)
        elif weight_scheme == "peto":        w_j = S_prev[j]
        elif weight_scheme == "fh":          w_j = S_prev[j] ** rho * (1 - S_prev[j]) ** gamma if weight_scheme == "fh" else 1.0
        elif weight_scheme == "tarone-ware": w_j = math.sqrt(n_j)
        else: raise ValueError("unknown weight_scheme")
        # Fleming-Harrington needs S_prev; compute if not yet
        if weight_scheme == "fh" and 'S_prev' not in dir():
            S_prev = _pooled_km_at(times, events, event_times)
            w_j = S_prev[j] ** rho * (1 - S_prev[j]) ** gamma
        U += w_j * (d_Aj - E_Aj)
        V += w_j * w_j * var_j
    if V <= 0:
        return {"U": U, "Var": V, "chi_square": float("nan"), "df": 1,
                "p_value": float("nan"), "group_A": a_label, "group_B": labels[1],
                "weight_scheme": weight_scheme}
    chi2 = U * U / V
    return {"U": float(U), "Var": float(V),
            "chi_square": float(chi2), "df": 1,
            "p_value": float(stats.chi2.sf(chi2, 1)),
            "group_A": a_label.item() if hasattr(a_label, "item") else a_label,
            "group_B": labels[1].item() if hasattr(labels[1], "item") else labels[1],
            "weight_scheme": weight_scheme,
            "method": f"weighted log-rank test ({weight_scheme})"}


def stratified_log_rank_test(times, events, group, strata,
                              weight_scheme: str = "logrank") -> dict:
    """Sum log-rank U and V across strata, then a single chi-square (§11.7)."""
    strata = np.asarray(strata); U_total = 0.0; V_total = 0.0
    per_stratum = []
    for s in np.unique(strata):
        mask = strata == s
        if len(np.unique(group[mask])) != 2: continue
        r = log_rank_test(times[mask], events[mask], group[mask], weight_scheme)
        U_total += r["U"]; V_total += r["Var"]
        per_stratum.append({"stratum": s.item() if hasattr(s, "item") else s,
                             **{k: r[k] for k in ("U", "Var", "chi_square", "p_value")}})
    if V_total <= 0:
        return {"chi_square": float("nan"), "df": 1, "p_value": float("nan"),
                "per_stratum": per_stratum, "weight_scheme": weight_scheme}
    chi2 = U_total * U_total / V_total
    return {"U_total": U_total, "Var_total": V_total,
            "chi_square": float(chi2), "df": 1,
            "p_value": float(stats.chi2.sf(chi2, 1)),
            "per_stratum": per_stratum,
            "weight_scheme": weight_scheme,
            "method": f"stratified log-rank ({weight_scheme})"}


def library_versions(times, events, group):
    try:
        from lifelines.statistics import logrank_test
        r = logrank_test(times[group == 0], times[group == 1],
                         events[group == 0], events[group == 1])
        return {"lifelines logrank": {"chi2": float(r.test_statistic), "p": float(r.p_value)}}
    except Exception as ex:
        return {"lifelines (optional)": f"not available: {ex}"}


if __name__ == "__main__":
    rng = np.random.default_rng(7)
    n = 100
    group = rng.choice([0, 1], size=n)
    # Group 1 has higher hazard
    T_event = rng.exponential(np.where(group == 1, 1 / 0.3, 1 / 0.15), n)
    C_censor = rng.uniform(0, 10, n)
    times = np.minimum(T_event, C_censor)
    events = (T_event <= C_censor).astype(int)

    for scheme in ("logrank", "wilcoxon", "peto", "tarone-ware"):
        r = log_rank_test(times, events, group, weight_scheme=scheme)
        print(f"=== {scheme} ===")
        print(f"  chi2 = {r['chi_square']:.4f}, p = {r['p_value']:.4g}")

    # Stratified: pretend we have 3 age strata
    strata = rng.choice([0, 1, 2], size=n)
    print("\n=== Stratified log-rank (3 strata) ===")
    r = stratified_log_rank_test(times, events, group, strata)
    print(f"  chi2 = {r['chi_square']:.4f}, p = {r['p_value']:.4g}")

    print("\n--- library ---")
    for k, v in library_versions(times, events, group).items():
        print(f"  {k}: {v}")
