# Latent-space network model (Reference Sec 30.5)
# Native R via latentnet; Python via graspologic.
# Run with:  Rscript latent_space_network.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  latentnet                    -- Handcock-Raftery-Tantrum MCMC + latent cluster models\n")
  cat("  ergm.count                   -- count-valued edges latent space extensions\n")
  cat("Python:\n")
  cat("  graspologic                  -- Microsoft LDA / LSM implementations\n")
  cat("  networkx.generators.lsm      -- generative alternatives\n")
  cat("Refs: Hoff, P.D., Raftery, A.E. & Handcock, M.S. (2002) 'Latent space approaches\n")
  cat("      to social network analysis', JASA;\n")
  cat("      Handcock, M.S., Raftery, A.E. & Tantrum, J.M. (2007) 'Model-based clustering\n")
  cat("      for social networks', JRSS-A.\n")
}
