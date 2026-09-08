# Turnbull NPMLE for interval-censored survival (Reference Sec 11.30)
# Native R via icenReg / interval / survival; Python via lifelines / from-scratch.
# Run with:  Rscript turnbull_interval_censored.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  icenReg::ic_np           -- NPMLE for interval-censored data\n")
  cat("  interval::icfit           -- Turnbull EM plus permutation tests\n")
  cat("  survival::survfit(Surv(time1, time2, type = 'interval2'))\n")
  cat("  bicreg / TurnbullEstimator wrappers\n")
  cat("Python:\n")
  cat("  lifelines.KaplanMeierFitter -- exact + right censoring baseline\n")
  cat("  scikit-survival + custom EM for interval censoring\n")
  cat("  From-scratch (see turnbull_interval_censored.py)\n")
  cat("Refs: Turnbull, B.W. (1976) 'The empirical distribution function with\n")
  cat("      arbitrarily grouped, censored and truncated data', JRSS-B 38(3);\n")
  cat("      Sun, J. (2006) The Statistical Analysis of Interval-Censored\n")
  cat("      Failure Time Data, Springer.\n")
}
