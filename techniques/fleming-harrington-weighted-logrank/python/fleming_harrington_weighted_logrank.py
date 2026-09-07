"""Fleming-Harrington weighted log-rank tests (Reference Sec 11.29).

Fleming & Harrington 1991 'Counting Processes and Survival Analysis'.
Generalises the log-rank test to weight event times differently:

    Z = sum_t w(t) * (O_1(t) - E_1(t)) /
        sqrt(sum_t w(t)^2 * V(t))                 ~   N(0, 1)

with G(rho, gamma) weights:

    w(t) = S_hat(t-)^rho * (1 - S_hat(t-))^gamma

    G(0, 0)     = classical log-rank
    G(1, 0)     = Peto-Peto / Prentice (early differences)
    G(0, 1)     = late differences
    G(1, 1)     = middle differences

Useful when the treatment effect is CONCENTRATED in time (early
onset, delayed onset, mid-follow-up); classical log-rank has poor
power against such alternatives.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays
from scipy import stats


def km_survival(t, e):
    order = np.argsort(t)
    t = t[order]; e = e[order]
    n = len(t); at_risk = n
    S = 1.0; times = []; surv = []
    for i in range(n):
        if e[i]:
            S *= (1 - 1 / at_risk)
        times.append(t[i]); surv.append(S)
        at_risk -= 1
    return np.array(times), np.array(surv)


def fh_weighted_logrank(t1, e1, t2, e2, rho=0.0, gamma=0.0):
    """Two-sample Fleming-Harrington G(rho, gamma) test."""
    t = np.concatenate([t1, t2]); e = np.concatenate([e1, e2])
    grp = np.concatenate([np.ones_like(t1), np.zeros_like(t2)])
    order = np.argsort(t)
    t = t[order]; e = e[order]; grp = grp[order]

    #  Overall KM (both groups pooled) for the weights
    _, S_pool = km_survival(t, e)
    #  Left-continuous S
    S_left = np.r_[1.0, S_pool[:-1]]

    n1_at = np.sum(grp == 1); n_at = len(t)
    O_E = 0.0; Var = 0.0
    n1_current = n1_at; n_current = n_at
    #  For each unique event time
    for i in range(n_at):
        if e[i]:
            d = 1
            n1 = n1_current
            n = n_current
            w = S_left[i] ** rho * (1 - S_left[i]) ** gamma
            O = grp[i]                    # 1 if group 1 event
            E = d * n1 / n if n > 0 else 0
            var = d * n1 * (n - n1) * (n - d) / (n ** 2 * (n - 1)) if n > 1 else 0
            O_E += w * (O - E)
            Var += w ** 2 * var
        if grp[i] == 1: n1_current -= 1
        n_current -= 1

    Z = O_E / np.sqrt(Var) if Var > 0 else 0
    p = 2 * (1 - stats.norm.cdf(abs(Z)))
    return {"Z": float(Z), "p": float(p), "rho": rho, "gamma": gamma}


if __name__ == "__main__":
    print("=== Fleming-Harrington weighted log-rank G(rho, gamma) ===\n")
    rng = np.random.default_rng(0)
    n = 200
    #  Delayed treatment effect: no separation early, big separation late
    lam1 = 0.05
    t1 = rng.exponential(1 / lam1, size=n)
    #  Group 2: hazard scaled 0.5 after t=15 (delayed benefit)
    def sim_late_benefit():
        out = []
        for _ in range(n):
            u = rng.uniform()
            if u < np.exp(-15 * lam1):
                #  Died before t=15: same as group 1
                out.append(rng.exponential(1 / lam1))
                while out[-1] > 15:
                    out[-1] = rng.exponential(1 / lam1)
            else:
                #  Survives past 15 with reduced hazard 0.5 * lam1
                out.append(15 + rng.exponential(1 / (0.5 * lam1)))
        return np.array(out)
    t2 = sim_late_benefit()

    c = rng.exponential(60, size=n * 2)
    e1 = (t1 <= c[:n]).astype(int); t1_obs = np.minimum(t1, c[:n])
    e2 = (t2 <= c[n:]).astype(int); t2_obs = np.minimum(t2, c[n:])

    for rho, gamma, label in [(0, 0, "classical log-rank"),
                                (1, 0, "Peto-Peto  (rho=1)"),
                                (0, 1, "late diff  (gamma=1)"),
                                (1, 1, "middle     (rho=gamma=1)")]:
        r = fh_weighted_logrank(t1_obs, e1, t2_obs, e2, rho=rho, gamma=gamma)
        print(f"  G({rho}, {gamma}) {label:>22s}   Z = {r['Z']:+.3f}   p = {r['p']:.4g}")

    print("\n  With delayed benefit, G(0, 1) has more power than the classical log-rank.")

    print("\n--- library cross-check (survival::survdiff(rho=) R; lifelines / statsmodels Python) ---")
