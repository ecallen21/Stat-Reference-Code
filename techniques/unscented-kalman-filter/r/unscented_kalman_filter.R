# Unscented Kalman filter (Reference Sec 18.13)
# Native R via mvKf / dse; Python via filterpy.
# Run with:  Rscript unscented_kalman_filter.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  mvKf                     -- multivariate Kalman + UKF variants\n")
  cat("  ukf                      -- lightweight UKF wrapper\n")
  cat("  bssm                     -- Bayesian SSM library incl. non-Gaussian\n")
  cat("Python:\n")
  cat("  filterpy.kalman.UnscentedKalmanFilter\n")
  cat("  pyfilter / dapper (UKF variants embedded in DA toolkits)\n")
  cat("Refs: Julier, S.J. & Uhlmann, J.K. (1997) 'A new extension of the\n")
  cat("      Kalman filter to nonlinear systems', SPIE AeroSense; Wan, E.A. &\n")
  cat("      van der Merwe, R. (2000) 'The unscented Kalman filter for\n")
  cat("      nonlinear estimation', IEEE AS-SPCC.\n")
}
