# Plackett-Luce ranking model (Reference Sec 47.53)
# Native R via PlackettLuce; Python via choix / from-scratch.
# Run with:  Rscript plackett_luce_ranking.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  PlackettLuce::PlackettLuce  -- ties, ranker-level covariates, PL-trees\n")
  cat("  pmr::pl                     -- probability model for ranking data\n")
  cat("  hyper2                      -- Hyperdirichlet extension of PL\n")
  cat("Python:\n")
  cat("  choix                       -- BT / PL / RUM MLE\n")
  cat("  from-scratch                -- see plackett_luce_ranking.py\n")
  cat("Refs: Plackett (1975) Appl Stat 24(2); Luce (1959) 'Individual Choice\n")
  cat("      Behavior', Wiley; Hunter (2004) 'MM algorithms for generalized\n")
  cat("      Bradley-Terry models', Ann Stat 32(1).\n")
}
