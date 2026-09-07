# Importance sampling (Reference Sec 45.6)
# Native R via custom; Python numpy + scipy.
# Run with:  Rscript importance_sampling.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  custom (base R + stats::dnorm)   -- SNIS + ESS from scratch\n")
  cat("  loo::psis                          -- Pareto-smoothed IS (Vehtari-Gelman)\n")
  cat("  rstan / brms                        -- LOO via PSIS-LOO\n")
  cat("Python:\n")
  cat("  scipy.stats + numpy               -- SNIS + ESS\n")
  cat("  arviz.psislw                       -- Pareto-smoothed weights\n")
  cat("  numpyro                            -- built-in IS + reweighting\n")
  cat("Refs: Geweke (1989) 'Bayesian inference in econometric models using Monte Carlo\n")
  cat("      integration', Econometrica; Vehtari, Gelman & Gabry (2017) 'Practical\n")
  cat("      Bayesian model evaluation using LOO-CV and WAIC', Stat Comput.\n")
}
