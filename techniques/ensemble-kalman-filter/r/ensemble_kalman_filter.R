# Ensemble Kalman filter (EnKF) (Reference Sec 18.11)
# Native R via dart_r / dartR / mvKf; Python via filterpy / dapper.
# Run with:  Rscript ensemble_kalman_filter.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  dartR / dart_r          -- data assimilation research testbed (R port)\n")
  cat("  MFEnKF                  -- multi-fidelity EnKF variants\n")
  cat("  KFAS (Kalman)            -- linear-Gaussian baseline\n")
  cat("Python:\n")
  cat("  filterpy.kalman.EnsembleKalmanFilter\n")
  cat("  dapper                    -- DA Performance Evaluation eR (extensive DA toolbox)\n")
  cat("  numpy-based custom (see ensemble_kalman_filter.py)\n")
  cat("Refs: Evensen, G. (1994) 'Sequential data assimilation with a nonlinear\n")
  cat("      quasi-geostrophic model using Monte Carlo methods to forecast\n")
  cat("      error statistics', J Geophys Res 99; Evensen, G. (2003) 'The\n")
  cat("      ensemble Kalman filter: theoretical formulation and practical\n")
  cat("      implementation', Ocean Dynamics 53(4).\n")
}
