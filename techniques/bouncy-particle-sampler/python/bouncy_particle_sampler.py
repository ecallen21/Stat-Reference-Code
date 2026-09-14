"""Bouncy Particle Sampler (Bouchard-Cote, Vollmer & Doucet 2018 JASA).

Piecewise-deterministic MCMC. Particle moves in straight lines
at unit speed; direction reflects off the gradient of −log π
at random times drawn from a Poisson process with rate

    lambda(x, v) = max(0, <v, grad U(x)>)     where U = -log pi

Additional refresh rate lambda_ref keeps ergodicity.

Non-reversible → often better mixing than reversible MCMC.
"""

import numpy as np    # arrays + random


def sample_time_to_bounce(x, v, grad_U, upper_M, rng):
    """Poisson-thinning to sample first event time with rate lambda(t) <= upper_M."""
    t = 0.0
    while True:
        t = t + rng.exponential(1.0 / upper_M)
        x_new = x + t * v
        rate = max(0.0, v @ grad_U(x_new))
        if rng.uniform() < rate / upper_M:
            return t, x_new


def bps(grad_U, x0, n_events, ref_rate=1.0, upper_M=5.0, seed=0):
    rng = np.random.default_rng(seed)
    x = np.array(x0, dtype=float)
    v = rng.standard_normal(len(x))
    v = v / np.linalg.norm(v)
    trace = [x.copy()]
    times = [0.0]
    for _ in range(n_events):
        # time to next bounce
        t_bounce, x_bounce = sample_time_to_bounce(x, v, grad_U, upper_M, rng)
        # time to next refresh
        t_ref = rng.exponential(1.0 / ref_rate)
        if t_ref < t_bounce:
            x = x + t_ref * v
            v = rng.standard_normal(len(x))
            v = v / np.linalg.norm(v)
            times.append(times[-1] + t_ref)
        else:
            x = x_bounce
            g = grad_U(x)
            g_norm2 = g @ g
            if g_norm2 > 0:
                v = v - 2 * (v @ g) / g_norm2 * g    # elastic bounce
            times.append(times[-1] + t_bounce)
        trace.append(x.copy())
    return np.array(trace), np.array(times)


def demo():
    print("=== Bouncy Particle Sampler (Bouchard-Cote et al 2018 JASA) ===")
    d = 3
    Sigma = np.array([[2.0, 0.7, 0.3],
                      [0.7, 1.5, 0.4],
                      [0.3, 0.4, 1.0]])
    Sigma_inv = np.linalg.inv(Sigma)

    def grad_U(x):
        return Sigma_inv @ x    # gradient of 0.5 x.T Sigma_inv x

    trace, times = bps(grad_U, x0=np.zeros(d), n_events=5000,
                       ref_rate=1.0, upper_M=15.0, seed=1)
    # continuous-time samples via linear interp of trace at grid
    # here just use event states after burn-in as sample
    samples = trace[500:]
    emp_cov = np.cov(samples.T)
    err = np.linalg.norm(emp_cov - Sigma, "fro") / np.linalg.norm(Sigma, "fro")
    print(f"  Target: N(0, Sigma), d={d}")
    print(f"  Sample mean = {np.round(samples.mean(axis=0), 3)}  (target zeros)")
    print(f"  Sample cov Frobenius rel err = {err:.3f}")
    print(f"  Total simulated time = {times[-1]:.1f}")


if __name__ == "__main__":
    demo()
