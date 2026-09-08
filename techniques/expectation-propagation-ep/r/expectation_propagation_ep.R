# Expectation Propagation (Reference Sec 47.120)
# Limited R (EPGLM); Python via GPy / from-scratch.
# Run with:  Rscript expectation_propagation_ep.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  EPGLM                      -- EP for GLMs (Chopin & Ridgway 2015)\n")
  cat("  brms / stan                -- HMC as gold-standard alternative\n")
  cat("Python:\n")
  cat("  GPy.likelihoods + GPy.inference.EP -- EP for GP classification\n")
  cat("  pymc.stats.compute_log_likelihood  -- ADVI + EP-like initialisers\n")
  cat("  scikit-learn.gaussian_process       -- Laplace + EP for GPC\n")
  cat("  from-scratch                        -- see expectation_propagation_ep.py\n")
  cat("Refs: Minka (2001) 'Expectation Propagation', UAI; Rasmussen & Williams\n")
  cat("      (2006) 'Gaussian Processes for Machine Learning', MIT Press, ch 3.6.\n")
}
