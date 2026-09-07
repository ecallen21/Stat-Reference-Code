# Stochastic-gradient MCMC (SGLD / SGHMC) (Reference Sec 25.9)
# Native R via SGmcmc; Python via tfp / numpyro / custom.
# Run with:  Rscript stochastic_gradient_mcmc.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  SGmcmc                   -- SGLD / SGHMC / SGRLD / SGNHT variants\n")
  cat("  BayesianTools            -- DE-MCMC + AM (frequentist for large n)\n")
  cat("Python:\n")
  cat("  tensorflow-probability.mcmc.LangevinDynamics -- SGLD-style samplers\n")
  cat("  numpyro.infer.SVI + preconditioned SGLD\n")
  cat("  pymc + custom step (SGLD stepper)\n")
  cat("  torch-based deep-model SGMCMC (BayesianCNN, MC-Dropout)\n")
  cat("Refs: Welling, M. & Teh, Y.W. (2011) 'Bayesian learning via stochastic\n")
  cat("      gradient Langevin dynamics', ICML; Chen, T., Fox, E. & Guestrin, C.\n")
  cat("      (2014) 'Stochastic gradient Hamiltonian Monte Carlo', ICML; Teh,\n")
  cat("      Y.W., Thiery, A.H. & Vollmer, S.J. (2016) 'Consistency and\n")
  cat("      fluctuations for stochastic gradient Langevin dynamics', JMLR.\n")
}
