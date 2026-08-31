# Stochastic Block Model (Reference Sec 30.4)
# Native R via sbm / blockmodels; Python via graspologic.
# Run with:  Rscript stochastic_block_model.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  sbm                          -- Latouche-Robin variational SBM (undirected + bipartite)\n")
  cat("  blockmodels                   -- Gaussian / Bernoulli / Poisson SBM\n")
  cat("  latentnet                     -- MCMC-based SBM + latent-space models\n")
  cat("Python:\n")
  cat("  graspologic                   -- Microsoft SBM + latent-space + spectral tools\n")
  cat("  graph-tool                    -- Peixoto Bayesian SBM (huge networks)\n")
  cat("  networkx.algorithms.community -- adjacent community detection\n")
  cat("Refs: Nowicki, K. & Snijders, T.A.B. (2001) 'Estimation and prediction for\n")
  cat("      stochastic blockstructures', JASA;\n")
  cat("      Daudin, J.-J., Picard, F. & Robin, S. (2008) 'A mixture model for random\n")
  cat("      graphs', Stat & Comp (variational EM).\n")
}
