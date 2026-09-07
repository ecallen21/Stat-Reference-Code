# Reversible-jump MCMC (Reference Sec 25.6)
# Native R via rjmcmc; Python via pymc + custom.
# Run with:  Rscript reversible_jump_mcmc.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  rjmcmc                       -- Green-style RJ-MCMC engine\n")
  cat("  mcmcse / coda                -- diagnostics for the output chain\n")
  cat("  mcmcabn / BayesLogit          -- model-search-friendly Gibbs samplers\n")
  cat("  MCMCglmm                      -- Bayesian mixed-model sampler\n")
  cat("Python:\n")
  cat("  From-scratch numpy (see reversible_jump_mcmc.py)\n")
  cat("  pymc (custom RJ via `Metropolis` + dimension-matching functions)\n")
  cat("  arviz (posterior diagnostics on any chain)\n")
  cat("Refs: Green, P.J. (1995) 'Reversible jump Markov chain Monte Carlo\n")
  cat("      computation and Bayesian model determination', Biometrika 82(4);\n")
  cat("      Richardson, S. & Green, P.J. (1997) 'On Bayesian analysis of\n")
  cat("      mixtures with an unknown number of components', JRSS-B 59(4).\n")
}
