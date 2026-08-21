# Multidimensional Item Response Theory (Reference §22.x extra)
# R via mirt.
# Run with:  Rscript mirt_multidimensional_irt.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  mirt::mirt(data, model='F1 = 1-10; F2 = 11-20', itemtype='2PL')  -- M2PL by Bock-Aitkin EM\n")
  cat("  mirt::mirt(..., method='MHRM')                                    -- Metropolis-Hastings Robbins-Monro (>2 dims)\n")
  cat("  mirt::fscores(fit, method='EAP')                                  -- ability estimates\n")
  cat("  mirt::itemplot(fit, item=1, type='trace')                         -- item response surfaces\n")
  cat("  latent variable rotation:  mirt::summary(fit, rotate='oblimin')\n")
  cat("Bayesian: brms with cumulative + a * theta parameterisation; or edstan; or Stan directly.\n")
}
