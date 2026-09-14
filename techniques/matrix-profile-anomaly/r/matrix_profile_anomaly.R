# Matrix Profile (Reference Sec 47.290)
# Native R via tsmp; Python via stumpy / matrixprofile-ts / from-scratch.
# Run with:  Rscript matrix_profile_anomaly.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  tsmp                       -- comprehensive matrix profile toolkit\n")
  cat("  anomalize                  -- companion anomaly detection\n")
  cat("Python:\n")
  cat("  stumpy.stump / stumped / gpu_stump\n")
  cat("  matrixprofile-ts (community)\n")
  cat("  scamp (large-scale)\n")
  cat("  From-scratch numpy (see matrix_profile_anomaly.py)\n")
  cat("Refs: Yeh, C.-C.M., Zhu, Y., Ulanova, L., Begum, N., Ding, Y., Dau, H.A.,\n")
  cat("      Silva, D.F., Mueen, A. & Keogh, E. (2016) 'Matrix profile I:\n")
  cat("      all pairs similarity joins for time series', ICDM.\n")
}
