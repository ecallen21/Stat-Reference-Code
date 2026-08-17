# Item analysis (Reference §22.2)
# R via psych::score.items or CTT::itemAnalysis.
# Run with:  Rscript item_analysis.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  psych::score.items(Y) -- item difficulty, discrimination, r_pb\n")
  cat("  CTT::itemAnalysis(Y)  -- comprehensive CTT item stats + distractor\n")
  cat("  ShinyItemAnalysis     -- interactive dashboard\n")
}
