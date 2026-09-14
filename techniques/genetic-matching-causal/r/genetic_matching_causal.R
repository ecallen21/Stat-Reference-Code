# Genetic Matching (Reference Sec 47.327)
# Native R via Matching::GenMatch / MatchIt; Python via reticulate / from-scratch.
# Run with:  Rscript genetic_matching_causal.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  Matching::GenMatch              -- authoritative implementation\n")
  cat("  MatchIt::matchit(method='genetic') -- unified matching interface\n")
  cat("  rgenoud                         -- underlying genetic optimiser\n")
  cat("Python:\n")
  cat("  reticulate + Matching::GenMatch  -- the go-to route\n")
  cat("  causalinference / causalmatch (approximate variants)\n")
  cat("  From-scratch GA (see genetic_matching_causal.py)\n")
  cat("Refs: Diamond, A. & Sekhon, J.S. (2013) 'Genetic matching for estimating\n")
  cat("      causal effects: a general multivariate matching method for achieving\n")
  cat("      balance in observational studies', Review of Economics and\n")
  cat("      Statistics 95(3).\n")
}
