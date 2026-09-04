# Multiple metrics + FDR (Reference Sec 44.5)
# Native R via stats::p.adjust + qvalue; Python statsmodels + custom.
# Run with:  Rscript multiple_metrics_fdr.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  stats::p.adjust                  -- Bonferroni, Holm, BH, BY, FDR\n")
  cat("  qvalue                            -- Storey q-values\n")
  cat("  mutoss                            -- multiple testing procedures suite\n")
  cat("Python:\n")
  cat("  statsmodels.stats.multitest.multipletests -- Bonferroni, BH, Holm, BY\n")
  cat("  scipy.stats                        -- p-values feeding into correction\n")
  cat("Refs: Kohavi, Tang & Xu (2020) Trustworthy Online Controlled Experiments,\n")
  cat("      CUP, Ch 17; Benjamini & Hochberg (1995) 'Controlling the false discovery\n")
  cat("      rate', JRSS-B.\n")
}
