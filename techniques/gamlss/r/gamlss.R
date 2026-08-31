# GAMLSS (Reference Sec 33.6)
# Native R has the reference implementation.
# Run with:  Rscript gamlss.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  gamlss                       -- Rigby-Stasinopoulos reference (100+ families)\n")
  cat("  gamlss.dist, gamlss.add       -- distribution catalogue + splines / GAM smooths\n")
  cat("  brms (family = 'gaussian(sigma ~ x)')   -- Bayesian GAMLSS via Stan\n")
  cat("  bamlss                        -- Bayesian additive models for location, scale, shape\n")
  cat("Python:\n")
  cat("  pyGAM                         -- generalised additive models; add sigma smooth manually\n")
  cat("  ngboost                       -- gradient boosting for full predictive distributions\n")
  cat("  distfit / statsmodels         -- distribution-fitting utilities\n")
  cat("Refs: Rigby, R.A. & Stasinopoulos, D.M. (2005) 'Generalized additive models for\n")
  cat("      location, scale and shape', J R Stat Soc C.\n")
}
