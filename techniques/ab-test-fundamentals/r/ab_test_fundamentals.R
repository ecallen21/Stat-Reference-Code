# A/B test fundamentals (Reference Sec 44.1)
# Native R via stats + pwr; Python scipy + statsmodels.
# Run with:  Rscript ab_test_fundamentals.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  stats::t.test / prop.test / chisq.test -- classical two-sample tests\n")
  cat("  pwr::pwr.t.test / pwr.2p.test           -- power / sample size\n")
  cat("Python:\n")
  cat("  scipy.stats.ttest_ind (Welch)            -- two-sample t\n")
  cat("  scipy.stats.chi2_contingency             -- categorical outcome\n")
  cat("  statsmodels.stats.proportion.proportions_ztest\n")
  cat("Refs: Kohavi, Tang & Xu (2020) Trustworthy Online Controlled Experiments,\n")
  cat("      CUP; Kohavi et al. (2009) 'Controlled experiments on the web', DMKD.\n")
}
