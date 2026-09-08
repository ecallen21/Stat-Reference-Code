# Stein Variational Gradient Descent (Reference Sec 47.87)
# No first-class R port; Python via pyro / numpyro / torch.
# Run with:  Rscript stein_variational_gradient.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  no established SVGD in mainstream R\n")
  cat("  greta                       -- Stan-like Bayesian modelling; MCMC baseline\n")
  cat("Python:\n")
  cat("  pyro.infer.SVGD             -- Pyro's SVGD implementation\n")
  cat("  numpyro.contrib.einstein    -- SVGD + Stein mixtures for NumPyro\n")
  cat("  torch custom + auto-grad    -- straightforward to port original paper\n")
  cat("  from-scratch                -- see stein_variational_gradient.py\n")
  cat("Refs: Liu & Wang (2016) 'Stein Variational Gradient Descent', NeurIPS;\n")
  cat("      Gorham & Mackey (2015) 'Measuring sample quality with Stein's\n")
  cat("      method', NeurIPS.\n")
}
