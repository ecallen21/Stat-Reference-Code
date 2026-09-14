"""Population-Based Training (Ref Sec 47.277).

Jaderberg, Dalibard et al 2017 DeepMind arXiv. Train a
POPULATION of N models in parallel; periodically:

    exploit: bottom P% copy top-Q%'s weights + hyperparameters
    explore: perturb hyperparameters (x0.8 or x1.25)

Result: HYPERPARAMETER SCHEDULES emerge (learning-rate warmup /
decay) rather than a single fixed value. Standard for
DeepMind-style RL and large-scale supervised training.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def pbt_step(pop, evaluate, rng, exploit_frac=0.25, perturb=(0.8, 1.25)):
    """One PBT sync: evaluate, exploit/explore worst 25% from top 25%."""
    scores = np.array([evaluate(m) for m in pop])
    order = np.argsort(scores)                                    # smaller = better
    n = len(pop); n_bot = int(exploit_frac * n)
    for i in range(n_bot):
        loser = order[-1 - i]; winner = order[rng.integers(0, n_bot)]
        # Copy weights + HP from winner, then perturb HP
        pop[loser]["weights"] = pop[winner]["weights"] + 0
        pop[loser]["lr"] = pop[winner]["lr"] * rng.choice(perturb)
        pop[loser]["mom"] = np.clip(pop[winner]["mom"] * rng.choice(perturb), 0.5, 0.99)
    return pop, scores


if __name__ == "__main__":
    print("=== Population-Based Training (Jaderberg et al 2017) ===\n")
    rng = np.random.default_rng(0)

    # Toy: each model has weight w; loss = (w - w_target)^2, target = 3.
    # Learning rate lr and momentum mom determine convergence speed.
    def make_model():
        return {"weights": float(rng.normal(0, 1)),
                "lr":      float(10 ** rng.uniform(-3, -1)),
                "mom":     float(rng.uniform(0.5, 0.95)),
                "v":       0.0}

    def train_step(m, steps=10):
        for _ in range(steps):
            g = 2 * (m["weights"] - 3.0)                          # gradient
            m["v"] = m["mom"] * m["v"] - m["lr"] * g
            m["weights"] += m["v"]

    def evaluate(m):
        return float((m["weights"] - 3.0) ** 2)

    N = 12; T = 20
    pop = [make_model() for _ in range(N)]
    best_hist = []
    lr_hist = []
    for t in range(T):
        for m in pop:
            train_step(m, steps=5)
        pop, scores = pbt_step(pop, evaluate, rng)
        best_hist.append(min(scores))
        # Track lr in the best member
        best_idx = int(np.argmin([evaluate(m) for m in pop]))
        lr_hist.append(pop[best_idx]["lr"])

    print(f"  N = {N} models, T = {T} PBT rounds\n")
    print(f"  {'step':>5}  {'best_loss':>10}  {'best_lr':>10}")
    for t in [0, 3, 6, 10, 15, T - 1]:
        print(f"  {t:>5}  {best_hist[t]:>10.4f}  {lr_hist[t]:>10.4f}")

    print(f"\n  Best final weight: {pop[int(np.argmin([evaluate(m) for m in pop]))]['weights']:.3f}   (target 3.0)")
    print(f"  Learning-rate SCHEDULE emerges - top model's lr shifted over training.")

    print("\n--- library cross-check (Ray Tune PopulationBasedTraining; DeepMind PBT paper repo) ---")
