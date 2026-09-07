# Hamiltonian Monte Carlo / NUTS (Reference Sec 14.4)
# Native R via rstan / cmdstanr / brms; Python numpyro / pymc / blackjax.
# Run with:  Rscript hmc_nuts.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  rstan / cmdstanr                  -- Stan interface with NUTS\n")
  cat("  brms                              -- brms::brm wraps rstan\n")
  cat("  rethinking                        -- educational Stan wrapper\n")
  cat("  BayesianTools                     -- alternative MCMC toolkit\n")
  cat("Python:\n")
  cat("  numpyro / pymc                    -- NUTS sampler\n")
  cat("  blackjax / tensorflow-probability -- NUTS + HMC primitives\n")
  cat("  emcee                              -- ensemble MCMC (not HMC)\n")
  cat("Refs: Duane, Kennedy, Pendleton & Roweth (1987) 'Hybrid Monte Carlo',\n")
  cat("      Phys Lett B; Neal (2011) 'MCMC using Hamiltonian dynamics' in Handbook\n")
  cat("      of MCMC; Hoffman & Gelman (2014) 'The No-U-Turn Sampler', JMLR.\n")
}
