# Adaptive Metropolis (Haario 2001) (Reference Sec 25.8)
# Native R via adaptMCMC / MCMCpack; Python via pymc / custom.
# Run with:  Rscript adaptive_metropolis_haario.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  adaptMCMC::MCMC(..., adapt = TRUE)  -- Roberts-Rosenthal AM\n")
  cat("  MCMCpack                          -- MCMCmetrop1R uses adaptive proposal\n")
  cat("  BayesianTools                     -- DE-MCMC + AM variants\n")
  cat("  mcmc::metrop                       -- basic RWMH baseline\n")
  cat("Python:\n")
  cat("  pymc AdaptiveMetropolis + custom step (nutpie / pyMC internal AM)\n")
  cat("  emcee (Foreman-Mackey ensemble sampler, related idea)\n")
  cat("  From-scratch numpy (see adaptive_metropolis_haario.py)\n")
  cat("Refs: Haario, H., Saksman, E. & Tamminen, J. (2001) 'An adaptive\n")
  cat("      Metropolis algorithm', Bernoulli 7(2): 223-242; Roberts, G.O. &\n")
  cat("      Rosenthal, J.S. (2007) 'Coupling and ergodicity of adaptive Markov\n")
  cat("      chain Monte Carlo algorithms', J Appl Probab 44.\n")
}
