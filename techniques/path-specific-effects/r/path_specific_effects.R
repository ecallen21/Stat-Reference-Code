# Path-specific effects (Reference Sec 15.41)
# Native R via paths / medflex; Python via DoWhy.
# Run with:  Rscript path_specific_effects.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  paths::paths                 -- Zhou & Yamamoto PSE estimation (Bayesian)\n")
  cat("  medflex::neImpute            -- extended natural-effect model for mediators\n")
  cat("  lavaan::sem                  -- SEM path coefficients\n")
  cat("Python:\n")
  cat("  DoWhy (Microsoft)             -- do-calculus + identification for PSE\n")
  cat("  causallib.estimation          -- custom multi-mediator estimators\n")
  cat("Refs: Avin, C., Shpitser, I. & Pearl, J. (2005) 'Identifiability of path-\n")
  cat("      specific effects', IJCAI; VanderWeele, T.J. & Chiba, Y. (2014)\n")
  cat("      'Multiple mediators: sensitivity analysis for path-specific effects',\n")
  cat("      Epidemiology 25(1); Zhou & Yamamoto (2023) paths package.\n")
}
