# Exponential Random Graph Model (ERGM; Reference §24.5)
# R via the statnet suite.
# Run with:  Rscript ergm_exponential_random_graph.R

if (sys.nframe() == 0) {
  cat("R packages (statnet ecosystem):\n")
  cat("  library(ergm)                          -- the ergm() fitter\n")
  cat("  ergm(net ~ edges + triangles)          -- MCMC-MLE\n")
  cat("  ergm(net ~ edges + gwesp(0.25, fixed=TRUE))  -- curved ERGM (avoids degeneracy)\n")
  cat("  ergm.pseudolikelihood(net ~ ...)       -- fast pseudo-likelihood estimate\n")
  cat("  simulate(fit, nsim=1000)               -- goodness-of-fit sims\n")
  cat("  gof(fit)                               -- diagnostic plots\n")
}
