# Elicitation of priors (Reference Sec 23.20)
# Native R via SHELF (Sheffield); Python via custom.
# Run with:  Rscript elicitation_of_priors.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  SHELF                     -- Sheffield Elicitation Framework (O'Hagan)\n")
  cat("  rriskDistributions       -- percentile-based prior fitting\n")
  cat("  bayestestR::describe_prior -- quick summarisation of expert-fit priors\n")
  cat("  prevalence                -- specialised elicitation for prevalence priors\n")
  cat("Python:\n")
  cat("  pyshelf (unofficial)      -- Python port of core SHELF routines\n")
  cat("  scipy.stats + optimize    -- from-scratch percentile matching (this .py)\n")
  cat("  arviz.plots.plot_prior     -- once you have samples\n")
  cat("Refs: Kadane, J.B. et al. (1980) 'Interactive elicitation of opinion for a\n")
  cat("      normal linear model', JASA 75(372): 845-854; O'Hagan, A. et al.\n")
  cat("      (2006) Uncertain Judgements: Eliciting Experts' Probabilities, Wiley;\n")
  cat("      SHELF web resources (Oakley & O'Hagan, ongoing).\n")
}
