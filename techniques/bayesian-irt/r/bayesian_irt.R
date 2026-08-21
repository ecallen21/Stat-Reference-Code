# Bayesian IRT (Reference §22.x extra)
# R via brms, rstan, or edstan.
# Run with:  Rscript bayesian_irt.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  brms::brm(response ~ 1 + (1 | item) + (1 | person), family=bernoulli)  -- Rasch\n")
  cat("  brms::brm(response ~ ... + gr(person) + gr(item, dpar='mu'), ...)      -- 2PL\n")
  cat("  edstan::stan_rasch / stan_2pl / stan_gpcm                              -- pre-compiled Stan models\n")
  cat("  rstan / rstanarm  -- roll your own Stan for constrained models\n")
  cat("  mirt::mirt(..., method='Bayesian') / TAM::tam.mml.mfr  -- posterior modes and quadrature\n")
  cat("Python: pymc, numpyro, pyro.  Polya-Gamma augmentation via pypolyagamma.\n")
}
