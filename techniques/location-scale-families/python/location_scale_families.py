"""Location-scale family — X = mu + sigma Z, Z ~ standard rv.

Common members:
    Normal, Cauchy, Student-t, Laplace, logistic, Gumbel, uniform.

CDF:  F(x; mu, sigma) = F_0((x - mu) / sigma)
PDF:  f(x; mu, sigma) = (1 / sigma) f_0((x - mu) / sigma)

Fisher information is DIAGONAL in the invariant (mu, log sigma)
parametrisation for symmetric families. MLE = (mean, sd) for
Normal, (median, MAD-scaled) for Cauchy / Laplace.
"""

import numpy as np    # arrays + random


def sample_ls(z_sampler, mu, sigma, n, rng):
    return mu + sigma * z_sampler(rng, n)


def normal_z(rng, n):
    return rng.standard_normal(n)


def laplace_z(rng, n):
    return rng.laplace(0, 1, size=n)


def cauchy_z(rng, n):
    u = rng.uniform(size=n)
    return np.tan(np.pi * (u - 0.5))


def demo():
    print("=== Location-scale families ===")
    rng = np.random.default_rng(2026)

    for name, sampler, robust_est in [
        ("Normal", normal_z, "mean/sd"),
        ("Laplace", laplace_z, "median/MAD"),
        ("Cauchy", cauchy_z, "median/IQR"),
    ]:
        mu_true, sigma_true = 2.0, 1.5
        x = sample_ls(sampler, mu_true, sigma_true, n=10000, rng=rng)
        mean_ = x.mean()
        sd_ = x.std(ddof=0)
        med_ = np.median(x)
        mad_ = 1.4826 * np.median(np.abs(x - med_))
        iqr_ = (np.percentile(x, 75) - np.percentile(x, 25)) / 1.349
        print(f"\n  {name}: X = 2 + 1.5 Z  (true mu=2, sigma=1.5)")
        print(f"    mean       = {mean_:+.3f}  sd(0)  = {sd_:.3f}")
        print(f"    median     = {med_:+.3f}  MAD    = {mad_:.3f}")
        print(f"    IQR-scale  = {iqr_:.3f}          {'  <-- ' if robust_est != 'mean/sd' else '  '}"
              f"({robust_est} recommended)")

    print("\nSee also: robust-location-scale, cauchy-distribution,")
    print("          huber-m-estimator, tukey-biweight-m-estimator,")
    print("          hodges-lehmann, quantile-regression.")


if __name__ == "__main__":
    demo()
