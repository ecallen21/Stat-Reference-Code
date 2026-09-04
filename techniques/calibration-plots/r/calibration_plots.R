# Calibration plots (Reference Sec 39.19)
# Native R via rms::calibrate; Python custom + sklearn.
# Run with:  Rscript calibration_plots.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  rms::calibrate                 -- bootstrap-corrected calibration curve\n")
  cat("  rms::val.prob                  -- reports slope + intercept + ICI\n")
  cat("  CalibrationCurves::val.prob.ci.2 -- calibration + confidence belt\n")
  cat("  predtools::calibration_plot    -- decile + spline / LOESS overlays\n")
  cat("  pmcalibration                  -- calibration belts\n")
  cat("Python:\n")
  cat("  sklearn.calibration.calibration_curve, CalibrationDisplay\n")
  cat("  custom                         -- grouped + LOESS + ICI/E-max/E-90\n")
  cat("Refs: Van Calster et al. (2019) 'Calibration: the Achilles heel of predictive\n")
  cat("      analytics', BMC Medicine; Austin & Steyerberg (2019) 'The integrated\n")
  cat("      calibration index (ICI)', Stat Med.\n")
}
