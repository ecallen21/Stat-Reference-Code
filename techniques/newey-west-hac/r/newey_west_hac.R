# Newey-West HAC + cluster-robust SEs (Reference Sec 35.15)
# Native R via sandwich; Python via statsmodels.
# Run with:  Rscript newey_west_hac.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  sandwich::vcovHAC             -- HAC SEs (Newey-West / Andrews)\n")
  cat("  sandwich::vcovCL              -- cluster-robust (CR1, CR2, CR3)\n")
  cat("  clubSandwich                  -- small-sample CR2 SEs with SE-t adjustment\n")
  cat("  lmtest::coeftest              -- combine with new SEs\n")
  cat("Python:\n")
  cat("  statsmodels OLS.fit(cov_type='HAC', cov_kwds={'maxlags': L})\n")
  cat("  statsmodels OLS.fit(cov_type='cluster', cov_kwds={'groups': ...})\n")
  cat("  linearmodels                  -- cluster / kernel / driscoll-kraay SEs\n")
  cat("Refs: Newey, W.K. & West, K.D. (1987) 'A simple, positive semi-definite,\n")
  cat("      heteroskedasticity and autocorrelation consistent covariance matrix',\n")
  cat("      Econometrica; Cameron, A.C. & Miller, D.L. (2015) 'A practitioner's guide\n")
  cat("      to cluster-robust inference', J Human Resources.\n")
}
