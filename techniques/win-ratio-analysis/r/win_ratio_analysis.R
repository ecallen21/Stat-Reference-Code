# Win Ratio Analysis (Reference Sec 47.268)
# Native R via WWR / WINrat / hierBinom; Python via from-scratch.
# Run with:  Rscript win_ratio_analysis.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  WWR                      -- weighted win-ratio for stratified trials\n")
  cat("  WINrat                   -- Pocock win-ratio with variance\n")
  cat("  hierBinom / winRatioAnal -- extensions with time-to-event\n")
  cat("  survival + custom loop   -- straightforward pairwise comparisons\n")
  cat("Python:\n")
  cat("  From-scratch pairwise comparison (see win_ratio_analysis.py)\n")
  cat("  R-package call via reticulate\n")
  cat("Refs: Pocock, S.J., Ariti, C.A., Collier, T.J. & Wang, D. (2012)\n")
  cat("      'The win ratio: a new approach to the analysis of composite\n")
  cat("      endpoints in clinical trials based on clinical priorities',\n")
  cat("      Eur Heart J 33(2), 176-182.\n")
}
