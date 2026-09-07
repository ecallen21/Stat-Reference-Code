# Accelerated failure time (AFT) models (Reference Sec 11.25)
# Native R via survival::survreg; Python via lifelines.
# Run with:  Rscript accelerated_failure_time.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  survival::survreg        -- Weibull / lognormal / loglogistic AFT\n")
  cat("  flexsurv::flexsurvreg    -- broader distribution list (gompertz, gen-gamma)\n")
  cat("  eha::phreg / aftreg      -- alternative AFT / PH implementations\n")
  cat("  rms::psm                 -- parametric survival model within Harrell rms\n")
  cat("Python:\n")
  cat("  lifelines.WeibullAFTFitter / LogNormalAFTFitter / LogLogisticAFTFitter\n")
  cat("  sksurv.linear_model.CoxPHSurvivalAnalysis (semi-param baseline)\n")
  cat("Refs: Kalbfleisch, J.D. & Prentice, R.L. (2002) The Statistical Analysis of\n")
  cat("      Failure Time Data, 2nd ed., Wiley; Collett, D. (2015) Modelling\n")
  cat("      Survival Data in Medical Research, 3rd ed., CRC.\n")
}
