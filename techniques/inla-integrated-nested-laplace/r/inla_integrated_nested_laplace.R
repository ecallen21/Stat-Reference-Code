# INLA -- integrated nested Laplace approximation (Reference Sec 25.13)
# Native R via INLA (r-inla.org); Python via pymc / numpyro (approximation-only).
# Run with:  Rscript inla_integrated_nested_laplace.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  INLA (r-inla.org, not on CRAN)  -- reference implementation\n")
  cat("  inlabru                           -- point-process / SPDE extensions\n")
  cat("  brinla                            -- tutorial helpers for INLA\n")
  cat("  survival, BayesSurv: INLA is often called for spatial survival\n")
  cat("Python:\n")
  cat("  No pip-installable INLA port -- closest is pymc (Laplace + HMC)\n")
  cat("  From-scratch Laplace + grid over hyperparameters (see .py)\n")
  cat("  cmdstanpy / numpyro (custom Laplace + variational alternatives)\n")
  cat("Refs: Rue, H., Martino, S. & Chopin, N. (2009) 'Approximate Bayesian\n")
  cat("      inference for latent Gaussian models by using integrated nested\n")
  cat("      Laplace approximations', JRSS-B 71(2); Rue, Riebler et al. (2017)\n")
  cat("      'Bayesian computing with INLA: a review', Ann Rev Stat Applic.\n")
}
