# Efron-Stein inequality (Reference Sec 46.12)
# No first-class CRAN package; from-scratch in Python.
# Run with:  Rscript efron_stein_inequality.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  No dedicated CRAN package -- implement from scratch\n")
  cat("  Related: bootstrap (variance of jackknife estimator relates to ES)\n")
  cat("  Related: distr / distrEx (theoretical variance computations)\n")
  cat("Python:\n")
  cat("  From-scratch numpy (see accompanying efron_stein_inequality.py)\n")
  cat("Refs: Efron, B. & Stein, C. (1981) 'The jackknife estimate of variance',\n")
  cat("      Ann Stat 9(3): 586-596; Boucheron, Lugosi & Massart (2013)\n")
  cat("      Concentration Inequalities: A Nonasymptotic Theory of Independence,\n")
  cat("      OUP.\n")
}
