# Inverse-Gaussian GLM (Reference Sec 47.325)
# Native R via stats::glm(family=inverse.gaussian); Python via statsmodels / from-scratch.
# Run with:  Rscript inverse_gaussian_glm.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  stats::glm(family = inverse.gaussian(link='log'))\n")
  cat("  MASS::glm.nb (companion for count / IG variants)\n")
  cat("  gamlss (family = IG)\n")
  cat("Python:\n")
  cat("  statsmodels.genmod.families.InverseGaussian\n")
  cat("  scipy.stats.invgauss for the distribution\n")
  cat("  From-scratch IRLS (see inverse_gaussian_glm.py)\n")
  cat("Refs: Tweedie, M.C.K. (1957) 'Statistical properties of inverse\n")
  cat("      Gaussian distributions I / II', Ann Math Stat 28;\n")
  cat("      McCullagh, P. & Nelder, J.A. (1989) 'Generalized Linear Models',\n")
  cat("      2nd ed, Chapman & Hall, chap 8.\n")
}
