# Particle filter / sequential Monte Carlo (Reference Sec 18.10)
# Native R via pomp / nimble; Python via particles / pyfilter.
# Run with:  Rscript particle_filter_smc.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  pomp::pfilter               -- Ionides SMC for POMP models\n")
  cat("  nimbleSMC                    -- particle MCMC for NIMBLE\n")
  cat("  SMC                          -- classical bootstrap / auxiliary PF\n")
  cat("  KFAS::simSSM                -- alternative state-space with FFBS\n")
  cat("Python:\n")
  cat("  particles (N. Chopin)        -- SMC^2, iterated smoothing, PMMH\n")
  cat("  pyfilter                     -- torch-backed PF with variational bridges\n")
  cat("  pymc                         -- SMC sampler for static problems\n")
  cat("Refs: Gordon, Salmond & Smith (1993) 'Novel approach to nonlinear/non-\n")
  cat("      Gaussian Bayesian state estimation', IEE Proc F 140(2); Doucet,\n")
  cat("      de Freitas & Gordon (2001) Sequential Monte Carlo Methods in\n")
  cat("      Practice, Springer.\n")
}
