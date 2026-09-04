# Negative outcome controls + empirical calibration (Reference Sec 43.9)
# Native R via EmpiricalCalibration (OHDSI); Python custom + scipy.
# Run with:  Rscript negative_outcome_controls.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  EmpiricalCalibration::fitNull / calibrateP -- OHDSI empirical calibration\n")
  cat("  Cyclops                                       -- large-scale regularised regression\n")
  cat("  MethodEvaluation (OHDSI)                      -- systematic negative-control frameworks\n")
  cat("Python:\n")
  cat("  custom (scipy.optimize for null MLE + scipy.stats.norm for p)\n")
  cat("  OHDSI tools via REST API\n")
  cat("Refs: Schuemie et al. (2014) 'Interpreting observational studies: why empirical\n")
  cat("      calibration is needed', Stat Med; Lipsitch, Tchetgen Tchetgen & Cohen (2010)\n")
  cat("      'Negative controls: a tool for detecting confounding and bias in\n")
  cat("      observational studies', Epidemiology.\n")
}
