# Coarsened Exact Matching (Reference Sec 15.10)
# Native R via MatchIt / cem; Python custom.
# Run with:  Rscript coarsened_exact_matching.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  MatchIt::matchit(method='cem')   -- CEM with automatic breaks or user-supplied\n")
  cat("  cem                                -- original Iacus-King-Porro implementation\n")
  cat("  cobalt                             -- balance diagnostics post-CEM\n")
  cat("Python:\n")
  cat("  custom (numpy)                    -- discretise + exact match + weights\n")
  cat("  causalinference                    -- adjacent matching tools\n")
  cat("Refs: Iacus, King & Porro (2012) 'Causal inference without balance checking:\n")
  cat("      coarsened exact matching', Political Analysis.\n")
}
