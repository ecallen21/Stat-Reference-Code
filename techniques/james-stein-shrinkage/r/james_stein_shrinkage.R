# James-Stein shrinkage (Reference Sec 38.15)
# R has no dedicated JS package; corpcor::cov.shrink is the multivariate
# extension.  Python custom.
# Run with:  Rscript james_stein_shrinkage.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  corpcor::cov.shrink            -- Ledoit-Wolf / JS-style covariance shrinkage\n")
  cat("  (JS estimator itself is one-liner; no dedicated package)\n")
  cat("Python:\n")
  cat("  sklearn.covariance.ShrunkCovariance / LedoitWolf\n")
  cat("  custom                         -- positive-part JS estimator\n")
  cat("Refs: Stein, C. (1956) 'Inadmissibility of the usual estimator for the mean of\n")
  cat("      a multivariate normal distribution', Proc Berkeley Symp; James & Stein\n")
  cat("      (1961) 'Estimation with quadratic loss', Proc Berkeley Symp;\n")
  cat("      Efron, B. & Morris, C. (1973) JASA.\n")
}
