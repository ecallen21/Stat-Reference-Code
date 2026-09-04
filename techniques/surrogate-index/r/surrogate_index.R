# Surrogate index for long-term effects (Reference Sec 44.14)
# Native R via stats::lm + custom; Python sklearn + custom.
# Run with:  Rscript surrogate_index.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  stats::lm                        -- OLS for surrogate weighting\n")
  cat("  mediation                          -- surrogate validation (mediation analysis)\n")
  cat("Python:\n")
  cat("  sklearn.linear_model.LinearRegression -- surrogate model\n")
  cat("  statsmodels + custom               -- two-stage SEs\n")
  cat("Refs: Athey, Chetty, Imbens & Kang (2020) 'The surrogate index: combining\n")
  cat("      short-term proxies to estimate long-term treatment effects more rapidly\n")
  cat("      and precisely', NBER Working Paper; Hohnhold, O'Brien & Tang (2015)\n")
  cat("      'Focusing on the long-term: it's good for users and business', KDD.\n")
}
