# Rejection sampling (Reference Sec 47.45)
# Native R via ars / Runuran; Python via scipy.stats / from-scratch.
# Run with:  Rscript rejection_sampling.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  ars::ars              -- adaptive rejection sampling (Gilks-Wild 1992)\n")
  cat("  Runuran::pinv.new     -- universal PINV inversion, often faster\n")
  cat("  Rejection             -- simple envelope helpers\n")
  cat("Python:\n")
  cat("  scipy.stats.<dist>.rvs        -- built-in fast sampling for standard dists\n")
  cat("  from-scratch                  -- see rejection_sampling.py\n")
  cat("Refs: von Neumann (1951) NBS AMS 12; Gilks & Wild (1992) Appl Stat 41(2);\n")
  cat("      Devroye (1986) 'Non-Uniform Random Variate Generation', Springer.\n")
}
