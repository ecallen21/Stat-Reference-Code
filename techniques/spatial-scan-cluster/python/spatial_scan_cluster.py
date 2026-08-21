"""Kulldorff spatial scan statistic (Reference §23.13).

For each candidate circular window Z centred at a location with a growing
radius (up to <= 50% of total cases), compute the likelihood ratio under
Bernoulli or Poisson:

Poisson version:
    LLR(Z) = c_Z * log(c_Z / mu_Z) + (C - c_Z) * log((C - c_Z) / (C - mu_Z))
             * 1{c_Z / mu_Z > (C - c_Z) / (C - mu_Z)}

where c_Z = # cases in Z, mu_Z = expected cases in Z under uniform risk
(population_in_Z / total_population * C).

The window with the largest LLR is the **most likely cluster**.  A Monte-Carlo
p-value is obtained by permuting cases across locations (fixed populations).

Reference: SaTScan (Kulldorff 1997).
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

import math    # stdlib: scalar math (sqrt, log)

import numpy as np    # numerical arrays + linear algebra


def _poisson_llr(c_z, mu_z, C):
    if c_z <= 0 or c_z >= C:
        return 0.0
    ratio_in = c_z / mu_z
    ratio_out = (C - c_z) / (C - mu_z)
    if ratio_in <= ratio_out:               # not an *excess* cluster
        return 0.0
    return (c_z * math.log(ratio_in) + (C - c_z) * math.log(ratio_out))


def _scan_once(coords, cases, pop, max_frac=0.5):
    coords = np.asarray(coords, dtype=float)
    cases = np.asarray(cases, dtype=float)
    pop = np.asarray(pop, dtype=float)
    n = len(cases); C = float(cases.sum()); P = float(pop.sum())
    best = {"llr": -np.inf, "centre": None, "radius": None,
            "obs": None, "exp": None, "members": None}
    for i in range(n):
        d = np.sqrt(((coords - coords[i]) ** 2).sum(-1))
        order = np.argsort(d)
        cum_cases = np.cumsum(cases[order])
        cum_pop = np.cumsum(pop[order])
        for k in range(n):
            c_z = cum_cases[k]
            mu_z = C * cum_pop[k] / P
            if c_z > max_frac * C:
                break
            llr = _poisson_llr(c_z, mu_z, C)
            if llr > best["llr"]:
                best.update({"llr": llr, "centre": int(i),
                              "radius": float(d[order[k]]),
                              "obs": float(c_z), "exp": float(mu_z),
                              "members": order[: k + 1].tolist()})
    return best


def spatial_scan(coords, cases, pop, max_frac: float = 0.5,
                 n_sim: int = 199, seed: int = 0) -> dict:
    obs_best = _scan_once(coords, cases, pop, max_frac)
    C = int(np.sum(cases))
    P = float(np.sum(pop))
    p_locs = np.asarray(pop, dtype=float) / P
    rng = np.random.default_rng(seed)
    sim_llr = np.zeros(n_sim)
    for s in range(n_sim):
        sim_counts = rng.multinomial(C, p_locs)
        sim_llr[s] = _scan_once(coords, sim_counts, pop, max_frac)["llr"]
    pval = (1 + int(np.sum(sim_llr >= obs_best["llr"]))) / (n_sim + 1)
    return {**obs_best, "p_value": pval, "n_sim": n_sim,
            "method": "Kulldorff spatial scan (Poisson)"}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # 10 x 10 grid of population centres
    xs, ys = np.meshgrid(np.arange(10), np.arange(10))
    coords = np.column_stack([xs.ravel(), ys.ravel()])
    n = len(coords)
    pop = np.full(n, 1000.0)
    # baseline expected rate 0.01 (=> ~10 cases per cell)
    base = 0.01 * pop
    # embed a 3x3 hotspot at centre (4-5, 4-5) with 4x the rate
    hot_mask = (np.abs(coords[:, 0] - 4.5) <= 1.5) & (np.abs(coords[:, 1] - 4.5) <= 1.5)
    lam = base.copy(); lam[hot_mask] *= 4.0
    cases = rng.poisson(lam)

    res = spatial_scan(coords, cases, pop, max_frac=0.3, n_sim=99, seed=1)
    print(f"=== Kulldorff spatial scan ===")
    print(f"  centre     = coords[{res['centre']}] = {coords[res['centre']].tolist()}")
    print(f"  radius     = {res['radius']:.3f}")
    print(f"  observed   = {res['obs']:.0f}   expected = {res['exp']:.2f}")
    print(f"  LLR        = {res['llr']:.2f}")
    print(f"  p-value    = {res['p_value']:.3f}  ({res['n_sim']} MC sims)")
    print(f"  # cells in cluster: {len(res['members'])}")
    # check overlap with true hotspot
    true_hot_idx = set(np.where(hot_mask)[0].tolist())
    detected = set(res["members"])
    print(f"  true hotspot cells: {len(true_hot_idx)}; "
          f"detected ∩ true: {len(detected & true_hot_idx)}")

    print("\n--- library cross-check (R SpatialEpi::kulldorff or SaTScan) ---")
