# MM-estimators for robust regression (Reference §17.x extra)
# R via robustbase.
# Run with:  Rscript mm_estimators_robust.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  robustbase::lmrob(y ~ x1 + x2, data)              -- default MM with FAST-S init\n")
  cat("  robustbase::lmrob.control(setting='KS2014')       -- Koller-Stahel default (95% eff, robust)\n")
  cat("  robustbase::covMcd(x)                             -- MCD covariance (multivariate MM)\n")
  cat("  MASS::rlm(y ~ x, method='MM', init='lts')         -- older MM with LTS init\n")
  cat("  robust::lmRob(y ~ x)                              -- competitor with different tuning\n")
  cat("Python: statsmodels.robust.RLM(TukeyBiweight) needs a robust starting point (LTS or S).\n")
}
