"""Hyperband Multi-Fidelity Search (Ref Sec 47.275).

Li, Jamieson, DeSalvo, Rostamizadeh & Talwalkar 2018 JMLR.
Bandit-based hyperparameter optimisation that adaptively allocates
budget between exploration (many configs, low budget) and
exploitation (few configs, high budget). Runs successive-halving
brackets across a geometrically decreasing s:

    for s in S..0:
        n = ceil((S+1)/(s+1)) * eta^s      configs
        r = R * eta^{-s}                    starting budget
        run successive-halving(n, r, eta)

Provably competitive with any allocation strategy up to log factors.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def successive_halving(configs, r, eta, evaluate):
    """Bracket of successive halving. Returns final surviving (config, score)."""
    survivors = list(configs)
    budget = r
    while len(survivors) > 1:
        scores = [evaluate(c, budget) for c in survivors]
        keep = max(1, int(np.floor(len(survivors) / eta)))
        idx = np.argsort(scores)[:keep]                           # smaller loss = better
        survivors = [survivors[i] for i in idx]
        budget *= eta
    return survivors[0], evaluate(survivors[0], budget)


def hyperband(R, eta, sample_config, evaluate):
    """Outer Hyperband loop; returns dict of best configs per bracket + overall."""
    S = int(np.floor(np.log(R) / np.log(eta)))
    best = None; log = []
    for s in range(S, -1, -1):
        n = int(np.ceil((S + 1) / (s + 1) * eta ** s))
        r = R * eta ** (-s)
        configs = [sample_config() for _ in range(n)]
        cfg, score = successive_halving(configs, r, eta, evaluate)
        log.append((s, n, r, cfg, score))
        if best is None or score < best[1]: best = (cfg, score)
    return best, log


if __name__ == "__main__":
    print("=== Hyperband (Li et al 2018 JMLR) ===\n")
    rng = np.random.default_rng(0)

    # Toy: hyperparam = (lr, hidden). Loss at budget r decays with sqrt(r) toward a
    # config-specific asymptote. Best asymptote near lr = 0.01, hidden = 64.
    def sample_config():
        return {"lr": float(10 ** rng.uniform(-4, -1)),
                "hidden": int(rng.choice([16, 32, 64, 128, 256]))}

    def evaluate(cfg, budget):
        # Config-specific true loss + budget-dependent noise
        best_lr = 0.01; best_hidden = 64
        true_loss = 0.5 * abs(np.log10(cfg["lr"] / best_lr)) + \
                    0.2 * abs(np.log2(cfg["hidden"] / best_hidden))
        noise = rng.normal(0, 0.5 / np.sqrt(budget))
        return float(true_loss + noise)

    R = 81; eta = 3
    best, log = hyperband(R, eta, sample_config, evaluate)
    print(f"  R = {R}, eta = {eta}, {int(np.floor(np.log(R)/np.log(eta)))+1} brackets\n")
    print(f"  {'bracket s':>10}  {'n_configs':>10}  {'r_start':>8}  best_score")
    for s, n, r, cfg, score in log:
        print(f"  {s:>10}  {n:>10}  {r:>8.2f}  {score:.4f}    (lr={cfg['lr']:.4f}, h={cfg['hidden']})")

    print(f"\n  Overall best: lr = {best[0]['lr']:.4f}, hidden = {best[0]['hidden']}   loss = {best[1]:.4f}")
    print(f"  (target region: lr near 0.01, hidden near 64)")

    print("\n--- library cross-check (Optuna HyperbandPruner; Ray Tune HyperBandScheduler) ---")
