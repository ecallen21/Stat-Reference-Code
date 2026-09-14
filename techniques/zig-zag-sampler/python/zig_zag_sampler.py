"""Zig-Zag process sampler (Bierkens-Fearnhead-Roberts 2019).

Non-reversible PDMP where each coordinate has an independent
velocity in {-1, +1}. Coordinate i's velocity flips with rate:

    lambda_i(x, v) = max(0, v_i * d/dx_i U(x))    U = -log pi

Times to flip are drawn by Poisson thinning against an
upper bound. Simpler bounces than the bouncy-particle sampler
(coordinate-wise), often better on very high-dim targets.
"""

import numpy as np    # arrays + random


def zig_zag(grad_U, x0, upper_bounds, n_events, seed=0):
    rng = np.random.default_rng(seed)
    x = np.array(x0, dtype=float)
    d = len(x)
    v = rng.choice([-1.0, 1.0], size=d)
    traj = [x.copy()]
    for _ in range(n_events):
        # For each coord, sample time to flip via Poisson thinning
        candidate_times = np.empty(d)
        for i in range(d):
            t = 0.0
            while True:
                t = t + rng.exponential(1.0 / upper_bounds[i])
                rate = max(0.0, v[i] * grad_U(x + t * v)[i])
                if rng.uniform() < rate / upper_bounds[i]:
                    break
            candidate_times[i] = t
        j = int(np.argmin(candidate_times))
        t_flip = candidate_times[j]
        x = x + t_flip * v
        v[j] = -v[j]
        traj.append(x.copy())
    return np.array(traj)


def demo():
    print("=== Zig-Zag sampler (Bierkens-Fearnhead-Roberts 2019) ===")
    print("Target: independent standard N(0, 1) in d=5")
    d = 5

    def grad_U(x):
        return x

    upper = np.full(d, 4.0)
    traj = zig_zag(grad_U, np.zeros(d), upper, n_events=5000, seed=1)
    burn = traj[500:]
    m = burn.mean(axis=0)
    v = burn.var(axis=0)
    print(f"  Sample mean per coord = {np.round(m, 3)}  (target 0)")
    print(f"  Sample var  per coord = {np.round(v, 3)}  (target ~ 1)")


if __name__ == "__main__":
    demo()
