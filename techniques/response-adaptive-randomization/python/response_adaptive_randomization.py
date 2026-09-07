"""Response-adaptive randomization (RAR) (Reference Sec 44.13).

Wei & Durham 1978 (RSIHR); Berry et al. 2010 'Bayesian Adaptive
Methods for Clinical Trials'. Modifies future randomisation
probabilities based on ACCUMULATED outcomes so more participants
receive the arm that is performing better.

Common flavours:
    * Play-the-winner (Wei-Durham urn) -- non-parametric.
    * Randomized-play-the-winner (RPW).
    * Thompson sampling / Bayesian adaptive randomization (BAR):
        p_A^{next} = Pr(theta_A > theta_B | data_so_far).

Trade-off: fewer participants assigned to inferior arm (ethical
appeal) vs. lower power / larger sample size to reach a fixed
type-1 / type-2 error.

We simulate a two-arm binary-outcome trial with true rates 0.30
(A) and 0.45 (B), comparing:
    * Fixed 1:1 randomisation
    * Wei-Durham urn
    * Bayesian adaptive Thompson randomisation
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def simulate_trial(true_p, N, scheme="fixed", rng=None, block=50):
    """Simulate one trial. Returns (n_assigned_A, n_events_A, n_assigned_B, n_events_B)."""
    rng = rng or np.random.default_rng()
    n_a = n_b = ev_a = ev_b = 0
    #  Urn state for Wei-Durham
    urn_a = urn_b = 1
    #  Priors for Thompson
    for i in range(N):
        if scheme == "fixed":
            arm = "A" if rng.uniform() < 0.5 else "B"
        elif scheme == "wei_durham":
            p = urn_a / (urn_a + urn_b)
            arm = "A" if rng.uniform() < p else "B"
        elif scheme == "thompson":
            #  Draw one sample from posterior Beta(1 + ev, 1 + n - ev)
            sa = rng.beta(1 + ev_a, 1 + n_a - ev_a)
            sb = rng.beta(1 + ev_b, 1 + n_b - ev_b)
            arm = "A" if sa > sb else "B"
        else:
            raise ValueError(scheme)
        y = int(rng.uniform() < (true_p[0] if arm == "A" else true_p[1]))
        if arm == "A":
            n_a += 1; ev_a += y
            if scheme == "wei_durham":
                urn_a += y; urn_b += 1 - y
        else:
            n_b += 1; ev_b += y
            if scheme == "wei_durham":
                urn_b += y; urn_a += 1 - y
    return n_a, ev_a, n_b, ev_b


if __name__ == "__main__":
    print("=== Response-adaptive randomization ===\n")
    rng = np.random.default_rng(0)
    true_p = (0.30, 0.45)         # arm B is better
    N = 400
    n_sim = 300
    print(f"  Two-arm trial: true P(success | A) = {true_p[0]}, P(success | B) = {true_p[1]}, N per trial = {N}\n")

    results = {}
    for scheme in ["fixed", "wei_durham", "thompson"]:
        n_a_arr = np.zeros(n_sim); ev_a_arr = np.zeros(n_sim)
        n_b_arr = np.zeros(n_sim); ev_b_arr = np.zeros(n_sim)
        rej_H0 = 0
        for k in range(n_sim):
            na, ea, nb, eb = simulate_trial(true_p, N, scheme, rng=rng)
            n_a_arr[k] = na; ev_a_arr[k] = ea
            n_b_arr[k] = nb; ev_b_arr[k] = eb
            #  Two-sample z-test on rates
            if na < 5 or nb < 5:
                continue
            pa = ea / na; pb = eb / nb
            se = np.sqrt(pa * (1 - pa) / na + pb * (1 - pb) / nb)
            z = (pb - pa) / se if se > 0 else 0
            if z > 1.96:
                rej_H0 += 1
        avg_n_b_pct = float(np.mean(n_b_arr) / N)
        avg_n_a_pct = float(np.mean(n_a_arr) / N)
        power = rej_H0 / n_sim
        results[scheme] = (avg_n_a_pct, avg_n_b_pct, power)
        print(f"  {scheme:12s}  %A={avg_n_a_pct * 100:.1f}%  %B={avg_n_b_pct * 100:.1f}%  "
              f"power(reject H0) = {power:.3f}")

    print("\n  Wei-Durham + Thompson steer patients toward B (better arm) but sacrifice")
    print("  some power vs 1:1 randomisation.")

    print("\n--- library cross-check (BAR / pipe R; adaptr / RAR pymc simulator Python) ---")
