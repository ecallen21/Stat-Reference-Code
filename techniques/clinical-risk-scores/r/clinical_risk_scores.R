# Clinical risk scores (Reference Sec 39.12)
# Native R via rms::nomogram + custom Sullivan integer-point rounding.
# Run with:  Rscript clinical_risk_scores.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  rms::nomogram + rms::points.chart -- convert fitted lrm/cph to points\n")
  cat("  AutoScore                        -- ML-based automated integer-score design\n")
  cat("  pmsampsize                       -- sample-size planning for the underlying model\n")
  cat("Python:\n")
  cat("  AutoScore (via rpy2)             -- automated scoring\n")
  cat("  custom                           -- Sullivan et al. 2004 integer-points method\n")
  cat("Refs: Sullivan, Massaro & D'Agostino (2004) 'Presentation of multivariate data\n")
  cat("      for clinical use: the Framingham Study risk score functions', Stat Med;\n")
  cat("      Moons et al. (2009) 'Prognosis and prognostic research: what, why, and\n")
  cat("      how?', BMJ; Xie, F. et al. (2020) 'AutoScore', JMIR Med Inform.\n")
}
