# Tweedie GLM (Reference Sec 47.323)
# Native R via statmod / tweedie / cplm; Python via statsmodels / from-scratch.
# Run with:  Rscript tweedie_glm_regression.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  statmod::tweedie              -- family for use in stats::glm\n")
  cat("  tweedie                        -- density evaluation + MLE\n")
  cat("  cplm                           -- compound-Poisson linear models\n")
  cat("  mgcv (Wood)                    -- Tweedie via gam family\n")
  cat("Python:\n")
  cat("  statsmodels.genmod.families.Tweedie\n")
  cat("  glm from statsmodels with power variance\n")
  cat("  From-scratch IRLS (see tweedie_glm_regression.py)\n")
  cat("Refs: Tweedie, M.C.K. (1984) 'An index which distinguishes between some\n")
  cat("      important exponential families', in Statistics: Applications and\n")
  cat("      New Directions;  Jorgensen, B. (1987) 'Exponential dispersion\n")
  cat("      models', JRSS-B 49.\n")
}
