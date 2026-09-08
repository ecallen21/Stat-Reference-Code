# RTS Kalman smoother (Reference Sec 47.127)
# Native R via KFAS / dlm; Python via filterpy / statsmodels.
# Run with:  Rscript rts_kalman_smoother.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  KFAS::KFS                        -- fast state-space + Kalman smoother\n")
  cat("  dlm::dlmSmooth                   -- Petris et al 2009 reference\n")
  cat("  MARSS                            -- multivariate autoregressive SS + smoother\n")
  cat("Python:\n")
  cat("  filterpy.kalman.rts_smoother     -- one-line RTS pass\n")
  cat("  statsmodels.tsa.statespace       -- KF + RTS via `smooth()`\n")
  cat("  pykalman                         -- classical KF / RTS / EM\n")
  cat("  from-scratch                     -- see rts_kalman_smoother.py\n")
  cat("Refs: Rauch, Tung & Striebel (1965) AIAA J 3(8); Sarkka (2013) 'Bayesian\n")
  cat("      Filtering and Smoothing', Cambridge.\n")
}
