# Histogram Binning Calibration (Reference Sec 47.285)
# Native R via probably / custom; Python via netcal / sklearn / from-scratch.
# Run with:  Rscript histogram_binning_calibration.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  probably::cal_estimate_isotonic  -- isotonic (monotone histogram)\n")
  cat("  custom cut() + tapply mean(y)    -- 5-line adaptive binning\n")
  cat("Python:\n")
  cat("  netcal.binning.HistogramBinning\n")
  cat("  sklearn.calibration.CalibratedClassifierCV(method='isotonic')\n")
  cat("  From-scratch numpy (see histogram_binning_calibration.py)\n")
  cat("Refs: Zadrozny, B. & Elkan, C. (2001) 'Obtaining calibrated probability\n")
  cat("      estimates from decision trees and naive Bayesian classifiers', ICML;\n")
  cat("      Zadrozny, B. & Elkan, C. (2002) 'Transforming classifier scores\n")
  cat("      into accurate multiclass probability estimates', KDD.\n")
}
