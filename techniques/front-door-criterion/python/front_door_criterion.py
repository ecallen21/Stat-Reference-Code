"""Pearl front-door adjustment (Reference §15.x extra).

DAG:
    U (unmeasured) --> T (treatment)
    U             --> Y (outcome)
    T --> M (mediator) --> Y

Even though T and Y are confounded by U, the causal effect P(Y | do(T)) is
identified via the front-door formula:

    P(Y | do(T = t)) = sum_m  P(M = m | T = t)
                       * sum_{t'} P(Y | T = t', M = m) * P(T = t')

Requires:
  * M mediates the entire effect of T on Y (no direct T -> Y).
  * U does not affect M directly.
  * No unblocked back-door from M to Y.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

import numpy as np    # numerical arrays + linear algebra


def front_door_binary(T, M, Y) -> dict:
    """Front-door estimand for BINARY T, M, Y."""
    T = np.asarray(T, dtype=int); M = np.asarray(M, dtype=int); Y = np.asarray(Y, dtype=int)
    n = len(T)
    # P(T = t)
    pT = np.array([(T == 0).mean(), (T == 1).mean()])
    # P(M = m | T = t)
    pM_T = np.zeros((2, 2))
    for t in (0, 1):
        idx = T == t
        pM_T[0, t] = (M[idx] == 0).mean()
        pM_T[1, t] = (M[idx] == 1).mean()
    # P(Y = 1 | T = t, M = m)
    pY_TM = np.zeros((2, 2))
    for t in (0, 1):
        for m in (0, 1):
            idx = (T == t) & (M == m)
            pY_TM[t, m] = Y[idx].mean() if idx.sum() > 0 else 0.5

    def _E_Y_do_T(t):
        s = 0.0
        for m in (0, 1):
            inner = sum(pY_TM[tp, m] * pT[tp] for tp in (0, 1))
            s += pM_T[m, t] * inner
        return s

    return {"E_Y_do_T0": float(_E_Y_do_T(0)),
            "E_Y_do_T1": float(_E_Y_do_T(1)),
            "ATE": float(_E_Y_do_T(1) - _E_Y_do_T(0)),
            "naive_diff": float(Y[T == 1].mean() - Y[T == 0].mean()),
            "method": "Pearl front-door adjustment (binary T, M, Y)"}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 20000
    # unmeasured confounder U ~ Bernoulli(0.5)
    U = rng.integers(0, 2, n)
    # T depends on U (higher prob if U=1)
    pT = np.where(U == 1, 0.8, 0.2)
    T = (rng.uniform(size=n) < pT).astype(int)
    # M mediates T's effect on Y; no U -> M
    pM = np.where(T == 1, 0.7, 0.1)
    M = (rng.uniform(size=n) < pM).astype(int)
    # Y depends on M and U (but not directly on T)
    pY = np.where(M == 1, 0.7, 0.1) + np.where(U == 1, 0.15, -0.15)
    pY = np.clip(pY, 0.01, 0.99)
    Y = (rng.uniform(size=n) < pY).astype(int)

    # True causal effect via a do-simulation
    def _true_do(t_do, m=None):
        M_do = (rng.uniform(size=n) < (0.7 if t_do == 1 else 0.1)).astype(int)
        pY_do = np.where(M_do == 1, 0.7, 0.1) + np.where(U == 1, 0.15, -0.15)
        pY_do = np.clip(pY_do, 0.01, 0.99)
        return pY_do.mean()

    truth = _true_do(1) - _true_do(0)

    fd = front_door_binary(T, M, Y)
    print(f"=== Pearl front-door adjustment (n={n}) ===")
    print(f"  E[Y | do(T=0)]  = {fd['E_Y_do_T0']:.4f}")
    print(f"  E[Y | do(T=1)]  = {fd['E_Y_do_T1']:.4f}")
    print(f"  ATE (front-door)= {fd['ATE']:+.4f}")
    print(f"  naive E[Y|T=1] - E[Y|T=0] = {fd['naive_diff']:+.4f}   (biased by U)")
    print(f"  TRUE ATE via do-simulation = {truth:+.4f}")

    print("\n--- library cross-check (R dowhy / bnlearn / do-calculus manual) ---")
