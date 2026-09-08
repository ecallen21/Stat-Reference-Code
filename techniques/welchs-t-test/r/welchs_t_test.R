# Welch's t-test (Reference Sec 47.113)
# Native R via stats::t.test; Python via scipy.stats.
# Run with:  Rscript welchs_t_test.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  stats::t.test(x, y, var.equal=FALSE)   -- default is Welch (var.equal=FALSE)\n")
  cat("  BSDA / afex                             -- wrappers with reporting helpers\n")
  cat("  effectsize::cohens_d                    -- effect-size complement\n")
  cat("Python:\n")
  cat("  scipy.stats.ttest_ind(a, b, equal_var=False)\n")
  cat("  pingouin.ttest(a, b, correction='auto') -- reports both + effect size\n")
  cat("  statsmodels.stats.weightstats.ttest_ind\n")
  cat("  from-scratch                            -- see welchs_t_test.py\n")
  cat("Refs: Welch (1947) Biometrika 34(1-2); Delacre, Lakens & Leys (2017) IRSP.\n")
}
