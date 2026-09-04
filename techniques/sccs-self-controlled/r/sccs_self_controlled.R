# Self-controlled case series (Reference Sec 43.2)
# Native R via SCCS; Python custom + rpy2.
# Run with:  Rscript sccs_self_controlled.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  SCCS::standardsccs / semisccs   -- classical and semi-parametric SCCS\n")
  cat("  gnm                              -- generalised nonlinear models with elim= for conditional Poisson\n")
  cat("  SelfControlledCaseSeries (OHDSI) -- distributed SCCS across CDMs\n")
  cat("Python:\n")
  cat("  sccs (via rpy2)                  -- R SCCS package proxy\n")
  cat("  custom (conditional Poisson 2-window)\n")
  cat("Refs: Farrington, C.P. (1995) 'Relative incidence estimation from case series\n")
  cat("      for vaccine safety evaluation', Biometrics; Petersen, Douglas & Whitaker\n")
  cat("      (2016) 'Self controlled case series methods: an alternative to standard\n")
  cat("      epidemiological study designs', BMJ.\n")
}
