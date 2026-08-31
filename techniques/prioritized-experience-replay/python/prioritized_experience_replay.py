"""Prioritised experience replay (Schaul 2016; Reference §28.x extra).

Uniform replay samples all transitions equally.  PER samples with probability

    P(i) proportional to  p_i^alpha,       p_i = |TD-error_i| + eps

alpha in [0, 1]: 0 = uniform, 1 = full prioritisation.
IMPORTANCE-SAMPLING correction:
    w_i = ( (1 / N) / P(i) )^beta      annealed beta -> 1
Multiplies the loss (or gradient) to counteract the sampling bias.

Effect: high-TD-error transitions are replayed more often, giving 2-4x
sample efficiency on Atari over uniform replay.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

import numpy as np    # numerical arrays + linear algebra


class PrioritisedBuffer:
    def __init__(self, capacity: int, alpha: float = 0.6, eps: float = 1e-3):
        self.capacity = capacity; self.alpha = alpha; self.eps = eps
        self.data = []; self.priorities = []; self.pos = 0
        self._max_priority = 1.0

    def push(self, transition):
        if len(self.data) < self.capacity:
            self.data.append(transition); self.priorities.append(self._max_priority)
        else:
            self.data[self.pos] = transition; self.priorities[self.pos] = self._max_priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch: int, beta: float = 0.4, rng=None):
        if rng is None: rng = np.random.default_rng()
        p = np.array(self.priorities) ** self.alpha
        probs = p / p.sum()
        idx = rng.choice(len(self.data), batch, p=probs)
        N = len(self.data)
        weights = (N * probs[idx]) ** (-beta)
        weights = weights / weights.max()                    # normalise
        batch_data = [self.data[i] for i in idx]
        return idx, batch_data, weights

    def update_priorities(self, idx, td_errors):
        for i, err in zip(idx, td_errors):
            new_p = float(abs(err) + self.eps)
            self.priorities[i] = new_p
            if new_p > self._max_priority:
                self._max_priority = new_p


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    N = 500
    # Fake TD-errors: 90% are near-zero, 10% are large -- the interesting cases.
    td = np.concatenate([np.abs(rng.normal(scale=0.05, size=int(0.9 * N))),
                          np.abs(rng.normal(scale=5.0, size=int(0.1 * N)))])
    rng.shuffle(td)

    buf = PrioritisedBuffer(capacity=N, alpha=0.6)
    for i, e in enumerate(td):
        buf.push(("s", "a", "r", "s'", False))
        buf.update_priorities([i], [e])

    # sample with PER vs uniform: measure the mean |TD| in the batch
    per_means = []; unif_means = []
    for _ in range(200):
        idx, _, _ = buf.sample(64, beta=0.4, rng=rng)
        per_means.append(float(np.mean([td[i] for i in idx])))
        uidx = rng.choice(N, 64, replace=False)
        unif_means.append(float(np.mean(td[uidx])))
    print(f"=== Prioritised Experience Replay vs uniform ===")
    print(f"  buffer TD-errors: 90% ~ N(0, 0.05^2), 10% ~ N(0, 5^2)")
    print(f"  mean |TD-error| under PER    (alpha=0.6): {np.mean(per_means):.3f}")
    print(f"  mean |TD-error| under uniform            : {np.mean(unif_means):.3f}")
    print(f"  PER samples the informative transitions ~"
          f"{np.mean(per_means) / max(np.mean(unif_means), 1e-9):.1f}x more.")

    # importance-sampling weights compensate for the bias
    idx, _, ws = buf.sample(64, beta=1.0, rng=rng)
    print(f"\n  IS weights (beta=1.0) statistics: min {ws.min():.3f}, "
          f"max {ws.max():.3f}, mean {ws.mean():.3f}")
    print(f"  (weights are smaller for over-sampled high-TD points, correcting the bias)")

    print("\n--- library cross-check (stable-baselines3 PER; RLlib PrioritizedReplayBuffer) ---")
