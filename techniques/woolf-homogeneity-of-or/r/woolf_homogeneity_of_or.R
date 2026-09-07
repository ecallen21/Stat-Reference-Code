# Woolf test of homogeneity of odds ratios (Reference Sec 4.17)
# Native R via DescTools / metafor; Python from-scratch.
# Run with:  Rscript woolf_homogeneity_of_or.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  DescTools::WoolfTest       -- classical Woolf statistic\n")
  cat("  metafor::rma.mh + fitstats  -- meta-analytic homogeneity\n")
  cat("  epitools::rateratio.wald    -- stratum-wise OR summary\n")
  cat("  stats::mantelhaen.test      -- CMH pooled OR baseline\n")
  cat("Python:\n")
  cat("  From-scratch scipy (see woolf_homogeneity_of_or.py)\n")
  cat("  statsmodels.stats.contingency_tables.StratifiedTable.test_equal_odds\n")
  cat("Refs: Woolf, B. (1955) 'On estimating the relationship between blood\n")
  cat("      group and disease', Ann Hum Genet 19(4): 251-253; Breslow, N.E.\n")
  cat("      & Day, N.E. (1980) Statistical Methods in Cancer Research Vol 1,\n")
  cat("      IARC (Breslow-Day companion test).\n")
}
