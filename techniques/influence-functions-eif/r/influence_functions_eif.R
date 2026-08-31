# Influence functions + EIF (Reference Sec 33.11)
# Native R for basic IF diagnostics; Python via reticulate.
# Run with:  Rscript influence_functions_eif.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  npcausal                     -- nonparametric causal EIFs\n")
  cat("  drtmle, tmle3                -- doubly-robust TMLE with EIF-based SEs\n")
  cat("  robustbase, MASS::rlm        -- influence-function-based robust estimators\n")
  cat("Python:\n")
  cat("  econml.dr, econml.iv          -- CATE / IV estimators with EIF-based SEs\n")
  cat("  dowhy                        -- causal DAG + influence-based estimation\n")
  cat("Refs: Hampel, F. (1974) 'The influence curve and its role in robust estimation', JASA;\n")
  cat("      Bickel, Klaassen, Ritov & Wellner (1993) 'Efficient and Adaptive Estimation\n")
  cat("      for Semiparametric Models', Johns Hopkins;\n")
  cat("      van der Vaart, A.W. (2000) 'Asymptotic Statistics', Cambridge U.P., Ch 25.\n")
}
