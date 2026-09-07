# Manski bounds -- partial identification (Reference Sec 15.34)
# Native R via bounds / relaxIV; Python via from-scratch or ivmodel.
# Run with:  Rscript manski_bounds.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  bounds                 -- Manski / Manski-Pepper bounds\n")
  cat("  relaxIV                -- weak-IV / partial-identification IV bounds\n")
  cat("  RATest                 -- randomization-based bounds\n")
  cat("Python:\n")
  cat("  From-scratch: E[Y|T=t], P(T=t), y_min, y_max are all you need\n")
  cat("  causalinference        -- causal inference helpers\n")
  cat("Refs: Manski, C.F. (1990) 'Nonparametric bounds on treatment effects',\n")
  cat("      AER 80(2): 319-323; Manski, C.F. (2003) Partial Identification\n")
  cat("      of Probability Distributions, Springer.\n")
}
