# Sample Size Re-estimation (Reference Sec 47.260)
# Native R via rpact / gsDesign / adaptTest; Python via scipy from-scratch.
# Run with:  Rscript sample_size_reestimation.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  rpact                    -- unblinded / blinded SSR workflows\n")
  cat("  gsDesign                 -- CHW weighted-Z + boundaries\n")
  cat("  adaptTest                -- adaptive one-sided testing\n")
  cat("  MAMS                     -- multi-arm multi-stage with SSR\n")
  cat("Python:\n")
  cat("  rpact via reticulate\n")
  cat("  From-scratch (see sample_size_reestimation.py)\n")
  cat("Refs: Wittes, J. & Brittain, E. (1990) 'The role of internal pilot\n")
  cat("      studies in increasing the efficiency of clinical trials',\n")
  cat("      Stat Med 9(1-2), 65-72;  Cui, L., Hung, H.M.J. & Wang, S.-J.\n")
  cat("      (1999) 'Modification of sample size in group sequential clinical\n")
  cat("      trials', Biometrics 55(3), 853-857.\n")
}
