# Nested sampling (Reference Sec 25.10)
# Native R via nestedmodels / Rnested; Python via dynesty / nestle / ultranest.
# Run with:  Rscript nested_sampling.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  nestedmodels           -- Skilling nested-sampling engine\n")
  cat("  RNested                -- Bayesian evidence via NS\n")
  cat("  BayesianTools          -- includes DE-MCMC with NS-style diagnostics\n")
  cat("Python:\n")
  cat("  dynesty                -- multi-modal / static / dynamic NS (recommended)\n")
  cat("  nestle                 -- Skilling reference implementation\n")
  cat("  ultranest              -- adaptive / MLFriends NS with parallelism\n")
  cat("  PolyChordLite / MultiNest -- gold-standard MultiNest variants\n")
  cat("Refs: Skilling, J. (2006) 'Nested sampling for general Bayesian\n")
  cat("      computation', Bayesian Anal 1(4): 833-859; Handley, Hobson &\n")
  cat("      Lasenby (2015) 'polychord', MNRAS.\n")
}
