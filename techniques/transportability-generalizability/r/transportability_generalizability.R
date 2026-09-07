# Transportability / generalizability (Reference Sec 15.39)
# Native R via generalize / transport; Python via causallib.
# Run with:  Rscript transportability_generalizability.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  generalize::transport      -- Westreich et al. IOW estimator\n")
  cat("  MatchThem                   -- multiply-imputed transport matching\n")
  cat("  txshift                    -- TMLE-based transported effects\n")
  cat("Python:\n")
  cat("  causallib.estimation.transport  -- IBM causallib transport module\n")
  cat("  DoWhy identification -- transportability node\n")
  cat("Refs: Cole, S.R. & Stuart, E.A. (2010) 'Generalizing evidence from RCTs\n")
  cat("      to target populations', Am J Epi 172(1); Westreich, D. et al.\n")
  cat("      (2017) 'Transportability of trial results using inverse-odds of\n")
  cat("      sampling weights', Am J Epi 186(8).\n")
}
