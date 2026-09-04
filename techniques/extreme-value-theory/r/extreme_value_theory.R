# Extreme value theory (Reference Sec 38.1)
# Native R via extRemes / evd; Python scipy.
# Run with:  Rscript extreme_value_theory.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  extRemes::fevd                 -- GEV + POT/GPD with return levels\n")
  cat("  evd                            -- densities + quantiles\n")
  cat("  evir, texmex                   -- alternative EVT toolboxes\n")
  cat("Python:\n")
  cat("  scipy.stats.genextreme         -- GEV MLE\n")
  cat("  scipy.stats.genpareto          -- GPD MLE\n")
  cat("  pyextremes (EVA)               -- high-level POT workflow\n")
  cat("Refs: Coles, S. (2001) An Introduction to Statistical Modeling of Extreme Values,\n")
  cat("      Springer; Beirlant, J. et al. (2004) Statistics of Extremes, Wiley.\n")
}
