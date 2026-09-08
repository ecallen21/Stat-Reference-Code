# Semi-Markov multi-state model (Reference Sec 47.47)
# Native R via SemiMarkov; Python is limited -- custom in demo.
# Run with:  Rscript semi_markov_multistate.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  SemiMarkov::semiMarkov   -- Weibull / exp / EW sojourn semi-Markov MLE\n")
  cat("  msm                      -- Markov multi-state, semi-Markov via panel\n")
  cat("  mstate                   -- Cox-Markov multi-state pipelines\n")
  cat("Python:\n")
  cat("  lifelines                -- univariate survival, no full semi-Markov\n")
  cat("  from-scratch             -- see semi_markov_multistate.py\n")
  cat("Refs: Foucher et al (2005) Biom J 47(6); Foucher et al (2010) 'Semi-Markov\n")
  cat("      model with covariates for AIDS epidemic', LIDA 16.\n")
}
