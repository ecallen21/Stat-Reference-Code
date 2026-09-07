# Cochran-Armitage trend test (Reference Sec 4.16)
# Native R via prop.trend.test / DescTools; Python via statsmodels.
# Run with:  Rscript cochran_armitage_trend.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  stats::prop.trend.test    -- built-in CAT with custom scores\n")
  cat("  DescTools::CochranArmitageTest -- fully-labelled output\n")
  cat("  coin::independence_test(teststat = 'quadratic') -- permutation variant\n")
  cat("Python:\n")
  cat("  statsmodels.stats.contingency_tables.Table2x2.oddsratio_pvalue?  -- no built-in\n")
  cat("  from-scratch scipy.stats.norm.cdf (see cochran_armitage_trend.py)\n")
  cat("Refs: Cochran, W.G. (1954) 'Some methods for strengthening the common\n")
  cat("      chi-squared tests', Biometrics 10; Armitage, P. (1955) 'Tests for\n")
  cat("      linear trends in proportions and frequencies', Biometrics 11.\n")
}
