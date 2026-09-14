"""Successive Halving / ASHA (Ref Sec 47.276).

Karnin, Koren & Somekh 2013 ICML (SH); Li, Jamieson et al 2020
MLSys (ASHA). Repeatedly evaluate n configs at budget r, keep
the top 1/eta, promote survivors to budget r*eta:

    Successive Halving (synchronous): wait for all n before halving
    ASHA (asynchronous):    promote whenever eta pass a rung

ASHA scales linearly across workers; a single-worker SH is a
special case. Used as the inner loop of Hyperband and BOHB.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def successive_halving(n, r, eta, sample_config, evaluate):
    """Synchronous SH. Returns best (config, score) and history per rung."""
    configs = [sample_config() for _ in range(n)]
    budget = r
    history = []
    while len(configs) > 1:
        scores = [evaluate(c, budget) for c in configs]
        history.append((budget, len(configs), min(scores)))
        keep = max(1, int(np.floor(len(configs) / eta)))
        idx = np.argsort(scores)[:keep]
        configs = [configs[i] for i in idx]
        budget *= eta
    return configs[0], evaluate(configs[0], budget), history


def asha(n_total, r_min, r_max, eta, sample_config, evaluate, rng):
    """Asynchronous SH: promote whenever eta configs pass a rung.
    Simplified synchronous-looking loop for illustration."""
    rung = {r_min: []}                                            # rung -> list of (score, cfg)
    n_started = 0
    while n_started < n_total:
        # Try to promote first
        promoted = False
        for r in sorted(rung.keys()):
            if len(rung[r]) >= eta:
                top = sorted(rung[r])[:len(rung[r]) // eta]
                r_next = r * eta
                if r_next > r_max: continue
                rung.setdefault(r_next, [])
                for score_c, cfg_c in top:
                    if cfg_c not in [c for _, c in rung[r_next]]:
                        s_new = evaluate(cfg_c, r_next)
                        rung[r_next].append((s_new, cfg_c))
                        promoted = True
                        break
                if promoted: break
        if not promoted:
            cfg = sample_config()
            rung[r_min].append((evaluate(cfg, r_min), cfg))
            n_started += 1
    # Best in highest rung reached
    best_r = max(r for r, lst in rung.items() if lst)
    best = sorted(rung[best_r])[0]
    return best[1], best[0], {r: len(lst) for r, lst in rung.items()}


if __name__ == "__main__":
    print("=== Successive Halving / ASHA (Karnin 2013; Li et al 2020) ===\n")
    rng = np.random.default_rng(0)

    def sample_config():
        return (float(10 ** rng.uniform(-4, -1)), int(rng.choice([16, 32, 64, 128])))

    def evaluate(cfg, budget):
        lr, hidden = cfg
        true_loss = 0.5 * abs(np.log10(lr / 0.01)) + 0.2 * abs(np.log2(hidden / 64))
        return float(true_loss + rng.normal(0, 0.5 / np.sqrt(budget)))

    print("  Successive Halving (n=27, r=1, eta=3):")
    (lr, h), score, hist = successive_halving(27, 1, 3, sample_config, evaluate)
    for r, k, best in hist:
        print(f"    budget = {r:>5.1f}   n_configs = {k:>2}   best-so-far loss = {best:.3f}")
    print(f"    final: lr={lr:.4f}, hidden={h}   loss={score:.4f}\n")

    print("  ASHA (n_total=40, r_min=1, r_max=27, eta=3):")
    (lr, h), score, rung_counts = asha(40, 1, 27, 3, sample_config, evaluate, rng)
    for r in sorted(rung_counts.keys()):
        print(f"    rung r = {r:>5.1f}   configs evaluated = {rung_counts[r]}")
    print(f"    best: lr={lr:.4f}, hidden={h}   loss={score:.4f}")

    print("\n  ASHA promotes IMMEDIATELY on eta rung entries -> lower wall-clock than SH,")
    print("  with same asymptotic budget efficiency.")

    print("\n--- library cross-check (Ray Tune ASHAScheduler; Optuna SuccessiveHalvingPruner) ---")
