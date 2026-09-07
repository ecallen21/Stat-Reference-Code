# Fisher's method for combining p-values (Reference Sec 22.16)
# Native R via metap / poolr; Python via scipy.stats.combine_pvalues.
# Run with:  Rscript fisher_combine_pvalues.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  metap::sumlog                 -- Fisher's chi-square sum-of-log-p test\n")
  cat("  metap::sumz                    -- Stouffer weighted-Z\n")
  cat("  metap::allmetap                -- Fisher / Stouffer / logit / Wilkinson\n")
  cat("  poolr::fisher / stouffer      -- corrections for correlated p\n")
  cat("  ACAT (Bioconductor)            -- Cauchy combination for GWAS\n")
  cat("Python:\n")
  cat("  scipy.stats.combine_pvalues(method = 'fisher'/'stouffer'/'pearson'/'tippett')\n")
  cat("  statsmodels.stats.combine_stats -- inverse-normal aggregation\n")
  cat("Refs: Fisher, R.A. (1932) Statistical Methods for Research Workers, 4th ed.,\n")
  cat("      Oliver & Boyd; Stouffer, S.A. et al. (1949) The American Soldier;\n")
  cat("      Liu, Y. & Xie, J. (2020) 'Cauchy combination test: a powerful test\n")
  cat("      with analytic p-value calculation under arbitrary dependency', JASA.\n")
}
