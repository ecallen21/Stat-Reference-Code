# Finite mixture of regressions (Reference Sec 36.3)
# Native R via flexmix / mixtools; Python custom.
# Run with:  Rscript mixture_regression.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  flexmix::flexmix                 -- canonical mixture-model framework\n")
  cat("  mixtools::regmixEM               -- EM for mixtures of regressions\n")
  cat("  poLCA (categorical only)          -- LCA baseline\n")
  cat("Python:\n")
  cat("  sklearn.mixture (Gaussian only)  -- baseline via covariance modelling\n")
  cat("  scikit-lego, custom EM            -- mixture regression\n")
  cat("Refs: DeSarbo & Cron (1988) 'A maximum likelihood methodology for clusterwise\n")
  cat("      linear regression', J Classification; Grun & Leisch (2007) 'Fitting\n")
  cat("      finite mixtures of generalized linear regressions in R', Comput Stat\n")
  cat("      Data Anal.\n")
}
