# Cox with time-varying covariates (Reference Sec 11.10)
# Native R via survival::coxph + tmerge; Python lifelines + custom.
# Run with:  Rscript cox_time_varying.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  survival::coxph + Surv(start, stop, event) + tmerge\n")
  cat("  timereg                            -- flexible time-varying-effect models\n")
  cat("Python:\n")
  cat("  lifelines::CoxTimeVaryingFitter   -- long-format time-varying Cox\n")
  cat("  scikit-survival                    -- basic Cox (add TV via long-format)\n")
  cat("  custom (partial-likelihood)\n")
  cat("Refs: Therneau & Grambsch (2000) Modeling Survival Data: Extending the Cox\n")
  cat("      Model, Springer; Fisher & Lin (1999) 'Time-dependent covariates in Cox\n")
  cat("      proportional hazards regression models', Annu Rev Public Health.\n")
}
