"""Continual Reassessment Method - CRM (Reference Sec 47.261).

O'Quigley, Pepe & Fisher 1990 Biometrics. Bayesian dose-finding
that updates a working DOSE-TOXICITY MODEL after each cohort
and treats the next patient at the dose whose posterior
toxicity is closest to the target rate p_target:

    p(d, a) = p_0(d) ^ exp(a)         (power model)
    a ~ Normal(0, sigma^2)             prior
    posterior: a | outcomes updated via Bayes; next dose picks
        argmin_d |E[p(d, a) | data] - p_target|

Adapts smoothly, uses ALL data, and estimates the MTD directly
(vs 3+3, which uses only the last cohort). Escalation with
Overdose Control (EWOC) adds a safety constraint.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def posterior_a(doses, tox, p_0, sigma_prior=1.4, n_grid=200):
    """Grid posterior of a in the power model p(d, a) = p_0[d]^exp(a)."""
    a_grid = np.linspace(-3, 3, n_grid)
    log_prior = -0.5 * (a_grid / sigma_prior) ** 2
    log_lik = np.zeros(n_grid)
    for d, t in zip(doses, tox):
        p_d = p_0[d] ** np.exp(a_grid)
        log_lik += t * np.log(p_d) + (1 - t) * np.log(1 - p_d)
    log_post = log_prior + log_lik
    log_post -= log_post.max()
    post = np.exp(log_post); post /= post.sum()
    return a_grid, post


def next_dose(p_0, a_grid, post, p_target):
    """Dose with posterior-mean toxicity closest to target."""
    p_mean = np.array([np.sum(post * p_0[d] ** np.exp(a_grid)) for d in range(len(p_0))])
    return int(np.argmin(np.abs(p_mean - p_target))), p_mean


if __name__ == "__main__":
    print("=== Continual Reassessment Method (O'Quigley et al 1990) ===\n")
    rng = np.random.default_rng(0)

    # 5 dose levels; skeleton p_0 = a priori guesses
    p_0 = np.array([0.05, 0.10, 0.20, 0.35, 0.50])
    # True toxicity curve (unknown to designer)
    p_true = np.array([0.02, 0.08, 0.18, 0.40, 0.65])
    p_target = 0.25

    doses, tox = [], []
    d = 1                                                          # start at dose 2 (0-indexed)
    print(f"  Dose skeleton: {p_0.tolist()}")
    print(f"  True tox:      {p_true.tolist()}")
    print(f"  Target = {p_target}, start at dose {d + 1}\n")

    print(f"  Cohort  Dose  Tox/n  Next-dose")
    for cohort in range(1, 11):
        # 3-patient cohort at current dose
        n = 3
        t_cohort = int(rng.binomial(n, p_true[d]))
        for _ in range(n):
            doses.append(d)
        for _ in range(t_cohort): tox.append(1)
        for _ in range(n - t_cohort): tox.append(0)
        a_grid, post = posterior_a(doses, tox, p_0)
        d_next, p_mean = next_dose(p_0, a_grid, post, p_target)
        print(f"  {cohort:>6}  {d + 1:>4}  {t_cohort}/{n}  {d_next + 1}   (post p ~ {p_mean.round(2).tolist()})")
        d = d_next

    # Recommended MTD after all cohorts
    print(f"\n  Recommended MTD after 10 cohorts: dose {d + 1} (true tox {p_true[d]:.2f})")

    print("\n  Compared to 3+3, CRM concentrates cohorts around the target dose")
    print("  and uses ALL data to estimate the MTD; typically escalates faster.")

    print("\n--- library cross-check (dfcrm R; bcrm R; UBCRM Python) ---")
