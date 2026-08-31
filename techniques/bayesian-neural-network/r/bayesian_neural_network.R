# Bayesian neural network (Reference Ch 29 Uncertainty Quantification)
# R via reticulate + Python — variational or MCMC inference.
# Run with:  Rscript bayesian_neural_network.R

if (sys.nframe() == 0) {
  cat("R packages: no first-class native R; use reticulate + Python.\n")
  cat("Python:\n")
  cat("  pyro                       -- pyro.nn.PyroModule, SVI, mean-field or normalising-flow\n")
  cat("  tensorflow-probability     -- tfp.layers.DenseVariational, DenseFlipout\n")
  cat("  bnn.torch, bayesian-torch  -- drop-in Bayesian layers for PyTorch\n")
  cat("  numpyro                    -- HMC / NUTS for full posterior sampling\n")
  cat("R alternatives:\n")
  cat("  brms + rstan               -- limited (Gaussian-process style), not deep BNNs\n")
  cat("  greta                      -- TF-Probability backend, small BNNs feasible\n")
  cat("Refs: Blundell et al. (2015) 'Weight Uncertainty in Neural Networks'\n")
  cat("      (Bayes by Backprop), ICML;\n")
  cat("      Kingma, Salimans & Welling (2015) 'Variational Dropout and the Local\n")
  cat("      Reparameterization Trick', NeurIPS.\n")
}
