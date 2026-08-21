# Spatial GLM (Reference §23.x extra)
# R via CARBayes, INLA, or spaMM.
# Run with:  Rscript spatial_glm.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  CARBayes::S.CARleroux(formula, family='poisson', W=W, burnin, n.sample)  -- Leroux CAR\n")
  cat("  CARBayes::S.CARbym(...)                                                    -- BYM (structured + unstructured)\n")
  cat("  INLA::inla(y ~ x + f(id, model='bym', graph=W_graph), family='poisson',\n")
  cat("             E=E, control.family=list(link='log'))                            -- fast Laplace\n")
  cat("  spaMM::HLCor(y ~ x + Matern(1 | x + y), family='poisson')                  -- Matern covariance\n")
  cat("  spdep::spglm                                                                -- GEE-based spatial GLM\n")
  cat("Python: pymc, numpyro (BYM in Stan/NumPyro), spreg::GM_Combo_Het for spatial-error probit.\n")
}
