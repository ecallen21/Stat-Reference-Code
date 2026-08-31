# Semiparametric efficiency (Reference Sec 33.4)
# Native R for the classic tools; Python via reticulate.
# Run with:  Rscript semiparametric_efficiency.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  tmle, tmle3                 -- targeted maximum-likelihood (efficient)\n")
  cat("  drtmle                       -- doubly robust TMLE\n")
  cat("  npcausal                     -- nonparametric causal-inference EIFs\n")
  cat("Python:\n")
  cat("  econml                       -- CATE / EIF-based estimators\n")
  cat("  dowhy                        -- causal DAG + estimation with EIF backends\n")
  cat("  zEpid                        -- epidemiological AIPW / TMLE\n")
  cat("Refs: Bickel, Klaassen, Ritov & Wellner (1993) 'Efficient and Adaptive\n")
  cat("      Estimation for Semiparametric Models', Johns Hopkins;\n")
  cat("      Tsiatis, A. (2006) 'Semiparametric Theory and Missing Data', Springer.\n")
}
