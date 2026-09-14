"""Sequential Monte Carlo samplers (Del Moral, Doucet & Jasra 2006).

Turn MCMC into a particle system by moving through a sequence
of intermediate distributions

    pi_0 (easy prior) --> pi_1 --> ... --> pi_T (target).

Each step:
    (1) MOVE particles via MH kernel with target pi_t
    (2) REWEIGHT by pi_{t+1} / pi_t
    (3) RESAMPLE if ESS too low.

Popular tempering: pi_t = prior^(1-lambda_t) * likelihood^lambda_t.
Also gives an UNBIASED estimate of the marginal likelihood.
"""

import numpy as np    # arrays + random


def smc_tempered(log_prior, log_lik, sample_prior, mh_step, n_particles,
                 lambda_schedule, seed=0):
    rng = np.random.default_rng(seed)
    X = sample_prior(rng, n_particles)
    log_w = np.zeros(n_particles)
    log_Z = 0.0
    T = len(lambda_schedule)
    for t in range(1, T):
        lam_prev, lam_curr = lambda_schedule[t - 1], lambda_schedule[t]
        # reweight: pi_{t+1} / pi_t proportional to exp((lam_curr - lam_prev) * log_lik)
        log_incr = (lam_curr - lam_prev) * np.array([log_lik(x) for x in X])
        log_w = log_w + log_incr
        # marginal likelihood contribution
        max_log = log_w.max()
        log_Z = log_Z + max_log + np.log(np.mean(np.exp(log_w - max_log)))
        log_w = log_w - max_log - np.log(np.mean(np.exp(log_w - max_log)))
        # ESS = 1 / sum(w^2) after normalisation
        w = np.exp(log_w - log_w.max())
        w = w / w.sum()
        ess = 1.0 / np.sum(w ** 2)
        # resample if ESS < N/2
        if ess < n_particles / 2:
            idx = rng.choice(n_particles, size=n_particles, p=w)
            X = X[idx]
            log_w = np.zeros(n_particles)
        # move via MH targeting current tempered posterior
        for i in range(n_particles):
            X[i] = mh_step(X[i], lam_curr, rng)
    return X, log_Z


def demo():
    print("=== SMC samplers (Del Moral-Doucet-Jasra 2006) ===")
    # Target: posterior on mean mu given data
    # prior: N(0, 4^2), likelihood: N(y | mu, 1)
    rng = np.random.default_rng(2026)
    y = rng.normal(3, 1.0, size=10)
    n = len(y)

    def log_prior(mu):
        return -0.5 * mu ** 2 / 16.0

    def log_lik(mu):
        return -0.5 * np.sum((y - mu) ** 2)

    def sample_prior(rng, N):
        return rng.normal(0, 4.0, size=N)

    def mh_step(mu, lam, rng):
        for _ in range(3):    # a few MH sweeps
            mu_new = mu + rng.normal(0, 0.5)
            log_alpha = log_prior(mu_new) + lam * log_lik(mu_new) \
                - log_prior(mu) - lam * log_lik(mu)
            if np.log(rng.uniform()) < log_alpha:
                mu = mu_new
        return mu

    lambda_schedule = np.linspace(0, 1, 20)
    X, log_Z_est = smc_tempered(log_prior, log_lik, sample_prior, mh_step,
                                 n_particles=1000,
                                 lambda_schedule=lambda_schedule, seed=1)

    # Analytical posterior for the conjugate Normal-Normal model:
    # posterior N(mu | (n * ybar) / (n + 1/16), 1 / (n + 1/16))
    prior_prec = 1 / 16.0
    lik_prec = n
    post_var = 1.0 / (prior_prec + lik_prec)
    post_mean = post_var * (n * y.mean())
    print(f"  True posterior N({post_mean:.3f}, {np.sqrt(post_var):.3f}^2)")
    print(f"  SMC empirical mean    = {X.mean():.3f}")
    print(f"  SMC empirical sd      = {X.std():.3f}")


if __name__ == "__main__":
    demo()
