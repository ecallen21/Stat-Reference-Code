"""Network diffusion / contagion models (Reference §24.8).

Discrete-time simulations of information / disease spread on a graph:

  * SI  (susceptible -> infected)                     — irreversible
  * SIR (susceptible -> infected -> recovered)         — with recovery
  * Independent cascade — each newly-active neighbour tries once with
    per-edge probability p to activate its neighbours.
  * Linear threshold — node activates when the fraction of active
    neighbours exceeds its threshold theta_i.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

import numpy as np    # numerical arrays + linear algebra


def si_simulate(A, seeds, beta: float, n_steps: int = 30, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed); A = np.asarray(A)
    n = A.shape[0]; state = np.zeros(n, dtype=int)
    state[list(seeds)] = 1
    infected_over_time = [state.sum()]
    for _ in range(n_steps):
        new_state = state.copy()
        infected = np.where(state == 1)[0]
        for u in infected:
            for v in np.where(A[u])[0]:
                if state[v] == 0 and rng.uniform() < beta:
                    new_state[v] = 1
        state = new_state; infected_over_time.append(state.sum())
        if state.sum() == n:
            break
    return {"final_infected": int(state.sum()),
            "trajectory": infected_over_time,
            "n_steps": len(infected_over_time) - 1,
            "method": "SI simulation"}


def sir_simulate(A, seeds, beta: float, gamma: float,
                 n_steps: int = 60, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed); A = np.asarray(A)
    n = A.shape[0]
    state = np.zeros(n, dtype=int)                    # 0=S, 1=I, 2=R
    state[list(seeds)] = 1
    S, I, R = [], [], []
    for _ in range(n_steps):
        S.append(int((state == 0).sum()))
        I.append(int((state == 1).sum()))
        R.append(int((state == 2).sum()))
        if I[-1] == 0:
            break
        new_state = state.copy()
        infected = np.where(state == 1)[0]
        for u in infected:
            for v in np.where(A[u])[0]:
                if state[v] == 0 and rng.uniform() < beta:
                    new_state[v] = 1
            if rng.uniform() < gamma:
                new_state[u] = 2
        state = new_state
    return {"S": S, "I": I, "R": R,
            "final_recovered": R[-1] if R else 0,
            "peak_infected": max(I) if I else 0,
            "method": "SIR simulation"}


def independent_cascade(A, seeds, p_edge: float, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed); A = np.asarray(A)
    n = A.shape[0]
    active = np.zeros(n, dtype=int); active[list(seeds)] = 1
    frontier = list(seeds)
    total = list(seeds)
    while frontier:
        new_frontier = []
        for u in frontier:
            for v in np.where(A[u])[0]:
                if not active[v] and rng.uniform() < p_edge:
                    active[v] = 1; new_frontier.append(v); total.append(v)
        frontier = new_frontier
    return {"final_active": int(active.sum()),
            "activation_order": total,
            "method": "independent-cascade"}


def linear_threshold(A, seeds, thresholds=None, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed); A = np.asarray(A, dtype=float)
    n = A.shape[0]
    if thresholds is None:
        thresholds = rng.uniform(0.1, 0.5, n)
    # normalise so weights out of each node sum to 1
    d = A.sum(axis=1); W = np.where(d[:, None] > 0, A / np.maximum(d[:, None], 1e-9), 0.0)
    active = np.zeros(n, dtype=int); active[list(seeds)] = 1
    for _ in range(n):                                  # at most n rounds
        influence = W.T @ active                        # incoming from active neighbours
        newly = (~active.astype(bool)) & (influence >= thresholds)
        if not newly.any():
            break
        active |= newly.astype(int)
    return {"final_active": int(active.sum()),
            "method": "linear-threshold"}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 100
    p = 0.05
    A = (rng.uniform(size=(n, n)) < p).astype(int)
    A = np.triu(A, 1); A = A + A.T
    seeds = [0, 1, 2]

    si = si_simulate(A, seeds, beta=0.15, n_steps=30, seed=1)
    print(f"=== SI (beta=0.15, seeds={seeds}) ===")
    print(f"  final infected = {si['final_infected']} / {n}  in {si['n_steps']} steps")

    sir = sir_simulate(A, seeds, beta=0.15, gamma=0.10, seed=1)
    print(f"\n=== SIR (beta=0.15, gamma=0.10) ===")
    print(f"  peak infected = {sir['peak_infected']}  "
          f"final recovered = {sir['final_recovered']}")

    ic = independent_cascade(A, seeds, p_edge=0.10, seed=1)
    print(f"\n=== Independent cascade (p_edge=0.10) ===")
    print(f"  final active = {ic['final_active']} / {n}")

    lt = linear_threshold(A, seeds, seed=1)
    print(f"\n=== Linear threshold (uniform(0.1, 0.5) thresholds) ===")
    print(f"  final active = {lt['final_active']} / {n}")

    print("\n--- library cross-check (ndlib / EoN) ---")
