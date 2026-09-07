# Extended Kalman filter (Reference Sec 18.12)
# Native R via KFAS + custom Jacobian; Python via filterpy.
# Run with:  Rscript extended_kalman_filter.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  KFAS::SSMcustom + numDeriv::jacobian  -- linear-Gaussian SSM + custom J\n")
  cat("  FKF                                    -- fast Kalman filter\n")
  cat("  bssm                                   -- Bayesian SSM (includes EKF)\n")
  cat("Python:\n")
  cat("  filterpy.kalman.ExtendedKalmanFilter\n")
  cat("  pykalman.KalmanFilter (linear baseline)\n")
  cat("  from-scratch numpy (see extended_kalman_filter.py)\n")
  cat("Refs: Jazwinski, A.H. (1970) Stochastic Processes and Filtering Theory,\n")
  cat("      Academic; Anderson & Moore (1979) Optimal Filtering, Prentice-Hall;\n")
  cat("      Simon, D. (2006) Optimal State Estimation, Wiley.\n")
}
