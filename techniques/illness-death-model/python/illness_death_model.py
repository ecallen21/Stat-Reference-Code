"""Illness-death multi-state model (Reference Sec 11.31).

Classical 3-state Markov model:

    (0) Healthy ---q_01---> (1) Ill ---q_12---> (2) Dead
         \\_______q_02___________________________^

with transition intensities q_01, q_02, q_12. State 2 (Dead) is
absorbing.

MLE from panel / continuous-time observation via the FORWARD
KOLMOGOROV equations:

    P'(t) = P(t) * Q     where P(0) = I, Q_ij = intensities, Q_ii = -sum_j Q_ij

Solution: P(t) = exp(Q * t). We fit the intensities from observed
sojourn times + transition counts (exact-time case).
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def fit_illness_death_exact(transitions):
    """Exact-time MLE. transitions: list of (from_state, to_state, sojourn_time)."""
    total_time = {0: 0.0, 1: 0.0}
    counts = {(0, 1): 0, (0, 2): 0, (1, 2): 0}
    for fs, ts, t in transitions:
        total_time[fs] += t
        if fs != ts:
            counts[(fs, ts)] += 1
    q_01 = counts[(0, 1)] / total_time[0]
    q_02 = counts[(0, 2)] / total_time[0]
    q_12 = counts[(1, 2)] / total_time[1]
    return {"q_01": q_01, "q_02": q_02, "q_12": q_12}


def transition_prob(Q, t):
    """P(t) = exp(Q t) via eigen decomposition."""
    from scipy.linalg import expm
    return expm(Q * t)


def simulate_illness_death(q_01, q_02, q_12, T_max, rng):
    """One subject's trajectory. Returns list of (from, to, sojourn)."""
    state = 0; t = 0.0; trans = []
    while state != 2 and t < T_max:
        if state == 0:
            rate = q_01 + q_02
            dt = rng.exponential(1 / rate)
            new = 1 if rng.uniform() < q_01 / rate else 2
        elif state == 1:
            dt = rng.exponential(1 / q_12)
            new = 2
        if t + dt > T_max:
            trans.append((state, state, T_max - t))
            break
        trans.append((state, new, dt))
        t += dt; state = new
    return trans


if __name__ == "__main__":
    print("=== Illness-death multi-state model ===\n")
    rng = np.random.default_rng(0)
    q_01_true, q_02_true, q_12_true = 0.10, 0.05, 0.15

    n = 400; T_max = 25
    all_trans = []
    for _ in range(n):
        all_trans.extend(simulate_illness_death(q_01_true, q_02_true, q_12_true, T_max, rng))

    r = fit_illness_death_exact(all_trans)
    print(f"  True (q_01, q_02, q_12) = ({q_01_true}, {q_02_true}, {q_12_true})")
    print(f"  Est  (q_01, q_02, q_12) = ({r['q_01']:.3f}, {r['q_02']:.3f}, {r['q_12']:.3f})")

    #  10-year transition probabilities
    Q = np.array([[-(q_01_true + q_02_true), q_01_true, q_02_true],
                    [0, -q_12_true, q_12_true],
                    [0, 0, 0]])
    P10 = transition_prob(Q, 10)
    print(f"\n  P(state at t = 10 | start = 0):  Healthy {P10[0, 0]:.3f}, "
          f"Ill {P10[0, 1]:.3f}, Dead {P10[0, 2]:.3f}")

    print("\n--- library cross-check (mstate / msm R; lifelines-multi-state Python) ---")
