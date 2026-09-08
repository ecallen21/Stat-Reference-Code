# Anderson-Darling GoF test (Reference Sec 47.101)
# Native R via nortest / goftest; Python via scipy.stats.
# Run with:  Rscript anderson_darling_test.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  nortest::ad.test           -- normality-focused Anderson-Darling\n")
  cat("  goftest::ad.test           -- general A-D for any parametric family\n")
  cat("  ADGofTest::ad.test         -- alternative implementation\n")
  cat("Python:\n")
  cat("  scipy.stats.anderson       -- tabulated cvalues for normal, expo, log-normal, etc\n")
  cat("  scipy.stats.anderson_ksamp -- k-sample AD\n")
  cat("  statsmodels                -- KS + AD wrappers\n")
  cat("  from-scratch               -- see anderson_darling_test.py\n")
  cat("Refs: Anderson & Darling (1954) JASA 49(268); D'Agostino & Stephens (1986)\n")
  cat("      'Goodness-of-Fit Techniques', Marcel Dekker.\n")
}
