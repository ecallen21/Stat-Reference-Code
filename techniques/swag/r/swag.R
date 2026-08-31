# SWAG - SWA-Gaussian (Reference Ch 29 Uncertainty Quantification)
# R via reticulate + Python (torch / pyro) - no canonical R package.
# Run with:  Rscript swag.R

if (sys.nframe() == 0) {
  cat("R packages: no canonical R package; use reticulate + Python.\n")
  cat("Python:\n")
  cat("  swag                       -- reference implementation (Maddox et al. 2019)\n")
  cat("  torchcontrib.optim.SWA     -- stochastic weight averaging optimiser\n")
  cat("  pyro / bayesian-torch      -- SWAG-flavoured posterior samplers\n")
  cat("R alternatives:\n")
  cat("  torch (R port)             -- collect optimiser iterates manually + fit Gaussian\n")
  cat("Refs: Maddox, W. et al. (2019) 'A Simple Baseline for Bayesian Uncertainty\n")
  cat("      in Deep Learning', NeurIPS.\n")
}
