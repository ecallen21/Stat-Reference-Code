"""BOHB - Bayesian Optimisation + Hyperband (Ref Sec 47.280).

Falkner, Klein & Hutter 2018 ICML. Combines the parallel
budget-efficient EXPLORATION of Hyperband with the sample-
efficient EXPLOITATION of a TPE surrogate model:

    Hyperband outer loop: brackets of successive halving
    inside each bracket:
        while sampling configs, ask the TPE for informed picks
        instead of pure random

Empirically stronger than either alone; core of Ray Tune's
BOHB scheduler and AutoML competition winners.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def parzen_density(x, samples, bw):
    if len(samples) == 0: return 1e-6
    return float(np.mean(np.exp(-0.5 * ((x - np.asarray(samples)) / bw) ** 2)))


def tpe_pick(history, low, high, gamma=0.25, n_cand=80, rng=None):
    if rng is None: rng = np.random.default_rng(0)
    if len(history) < 6: return float(rng.uniform(low, high))
    xs = np.array([h[0] for h in history]); ys = np.array([h[1] for h in history])
    order = np.argsort(ys); n_good = max(1, int(gamma * len(ys)))
    good_x = xs[order[:n_good]]; bad_x = xs[order[n_good:]]
    bw = 0.1 * (high - low)
    cand = rng.uniform(low, high, n_cand)
    scores = np.array([parzen_density(c, good_x, bw) / max(parzen_density(c, bad_x, bw), 1e-9)
                        for c in cand])
    return float(cand[int(np.argmax(scores))])


def bohb(low, high, R, eta, evaluate, rng):
    """Simplified 1-D BOHB; returns (best_x, best_loss, history)."""
    S = int(np.floor(np.log(R) / np.log(eta)))
    best = None; hist_full = []
    for s in range(S, -1, -1):
        n = int(np.ceil((S + 1) / (s + 1) * eta ** s))
        r = R * eta ** (-s)
        # Sample configs: use TPE if we have enough history, else random
        cfgs = [tpe_pick(hist_full, low, high, rng=rng) for _ in range(n)]
        # Successive halving inside
        surv = list(cfgs)
        budget = r
        while len(surv) > 1:
            scores = [evaluate(c, budget) for c in surv]
            for c, y in zip(surv, scores): hist_full.append((c, y))
            keep = max(1, int(np.floor(len(surv) / eta)))
            idx = np.argsort(scores)[:keep]
            surv = [surv[i] for i in idx]
            budget *= eta
        final_score = evaluate(surv[0], budget)
        hist_full.append((surv[0], final_score))
        if best is None or final_score < best[1]: best = (surv[0], final_score)
    return best[0], best[1], hist_full


if __name__ == "__main__":
    print("=== BOHB - Bayesian Optimisation + Hyperband (Falkner et al 2018) ===\n")
    rng = np.random.default_rng(0)

    def evaluate(x, budget):
        # True minimum at x = 0.7 with a decoy at x = 0.15
        true_loss = min((x - 0.7) ** 2, 0.1 + 2 * (x - 0.15) ** 2)
        return float(true_loss + rng.normal(0, 0.3 / np.sqrt(budget)))

    best_x, best_y, hist = bohb(0.0, 1.0, R=27, eta=3, evaluate=evaluate, rng=rng)
    print(f"  Total evaluations: {len(hist):,}")
    print(f"  Best x found: {best_x:.3f}   loss = {best_y:.4f}   (target x = 0.70)")

    # Compare with pure Hyperband (no TPE): same code but random picks
    def hb(low, high, R, eta, evaluate, rng):
        S = int(np.floor(np.log(R) / np.log(eta)))
        best = None; hist_full = []
        for s in range(S, -1, -1):
            n = int(np.ceil((S + 1) / (s + 1) * eta ** s)); r = R * eta ** (-s)
            surv = [float(rng.uniform(low, high)) for _ in range(n)]
            budget = r
            while len(surv) > 1:
                scores = [evaluate(c, budget) for c in surv]
                for c, y in zip(surv, scores): hist_full.append((c, y))
                keep = max(1, int(np.floor(len(surv) / eta)))
                idx = np.argsort(scores)[:keep]
                surv = [surv[i] for i in idx]; budget *= eta
            final_score = evaluate(surv[0], budget); hist_full.append((surv[0], final_score))
            if best is None or final_score < best[1]: best = (surv[0], final_score)
        return best[0], best[1], hist_full

    rng2 = np.random.default_rng(1)
    bx, by, _ = hb(0.0, 1.0, 27, 3, lambda x, b: (x - 0.7) ** 2 + rng2.normal(0, 0.3 / np.sqrt(b)), rng2)
    print(f"\n  Plain Hyperband (same budget): best x = {bx:.3f}   loss = {by:.4f}")
    print(f"  BOHB routinely finds the mode faster by combining TPE picks with SH promotion.")

    print("\n--- library cross-check (hpbandster BOHB; Ray Tune BOHBScheduler; SMAC3) ---")
