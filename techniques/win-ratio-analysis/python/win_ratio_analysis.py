"""Win Ratio Analysis (Reference Sec 47.268).

Pocock, Ariti, Collier & Wang 2012 Eur Heart J. HIERARCHICAL
COMPOSITE endpoint that ranks outcomes by clinical importance
(e.g., death > hospitalisation > quality of life). For every
treated-control pair, decide who "wins" using the highest-order
outcome that discriminates them, then compute:

    Win Ratio = (# treated wins) / (# control wins)

Reports the odds of a treatment patient having a better outcome
than a control patient. Popular in cardiovascular trials to
combine mortality with softer endpoints without the composite-
event-first flaw.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def win_ratio(t_outcomes, c_outcomes):
    """Pairwise hierarchical win-ratio for two arms with tiered outcomes.
    Each outcomes matrix has shape (n, K) where columns are ordered by importance.
    A LARGER value in column k is 'worse'. Compare on first tier that discriminates.
    """
    n_t, K = t_outcomes.shape
    n_c = c_outcomes.shape[0]
    wins_t = 0; wins_c = 0
    for i in range(n_t):
        for j in range(n_c):
            for k in range(K):
                if t_outcomes[i, k] < c_outcomes[j, k]:
                    wins_t += 1; break
                elif t_outcomes[i, k] > c_outcomes[j, k]:
                    wins_c += 1; break
                # else tied on this tier; move to next
    return wins_t, wins_c, wins_t / max(wins_c, 1)


def win_ratio_ci(wins_t, wins_c, n_t, n_c, level=0.95):
    """Bebu-Lachin variance approximation for log(WR)."""
    from scipy.stats import norm
    N = n_t * n_c
    p_t = wins_t / N; p_c = wins_c / N
    var_log_wr = 4 / N * (p_t * (1 - p_t) / (p_t + p_c) ** 2 + p_c * (1 - p_c) / (p_t + p_c) ** 2)
    z = norm.ppf(1 - (1 - level) / 2)
    log_wr = np.log(wins_t / max(wins_c, 1))
    return float(np.exp(log_wr - z * np.sqrt(var_log_wr))), float(np.exp(log_wr + z * np.sqrt(var_log_wr)))


if __name__ == "__main__":
    print("=== Win Ratio (Pocock, Ariti, Collier & Wang 2012 EHJ) ===\n")
    rng = np.random.default_rng(0)

    # Simulate: treatment reduces mortality slightly + hospitalisation more
    # Outcomes: (time-to-death, time-to-hosp, symptom-score)  smaller=better
    n = 300
    # Control
    death_c = rng.exponential(24, n)                              # median survival ~ 16 mo
    hosp_c  = np.minimum(rng.exponential(18, n), death_c)         # hospital before death
    symp_c  = rng.normal(50, 10, n)
    # Treated
    death_t = rng.exponential(30, n)
    hosp_t  = np.minimum(rng.exponential(30, n), death_t)
    symp_t  = rng.normal(45, 10, n)

    C = np.column_stack([-death_c, -hosp_c, symp_c])              # -death so "smaller is better"
    T = np.column_stack([-death_t, -hosp_t, symp_t])

    w_t, w_c, wr = win_ratio(T, C)
    lo, hi = win_ratio_ci(w_t, w_c, n, n)
    print(f"  N = {n} per arm, hierarchical outcomes: (death, hospitalisation, symptoms)")
    print(f"    treated wins: {w_t:,}")
    print(f"    control wins: {w_c:,}")
    print(f"    Win Ratio = {wr:.3f}    95% CI ({lo:.3f}, {hi:.3f})")
    print(f"    -> treatment {'preferred' if lo > 1 else 'not significantly preferred'}")

    # Contrast with a plain time-to-composite (any first event: death OR hospitalisation)
    first_event_c = np.minimum(death_c, hosp_c)
    first_event_t = np.minimum(death_t, hosp_t)
    from scipy.stats import ranksums
    z, p = ranksums(first_event_t, first_event_c)
    print(f"\n  Naive composite (time to first event) Wilcoxon: z = {z:.2f}, p = {p:.4f}")
    print(f"  Win ratio uses the HIERARCHY - a death loss can't be masked by a hospital 'win'.")

    print("\n--- library cross-check (WWR R; WINrat R; simple loop possible) ---")
