# Conditional mutual information + CI test (Reference Sec 34.13)
# Native R via bnlearn / condMI; Python via NPEET.
# Run with:  Rscript conditional_mutual_info.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  bnlearn::ci.test              -- MI / chi-square / permutation CI tests\n")
  cat("  condMI                        -- KSG-based continuous CMI\n")
  cat("  infotheo::condentropy         -- discrete conditional MI\n")
  cat("Python:\n")
  cat("  NPEET (KSG estimator)         -- continuous CMI\n")
  cat("  pgmpy.estimators.CITest       -- probabilistic-graph CI tests\n")
  cat("  causal-learn.CIT              -- Fisher-Z / KCI / RCIT tests\n")
  cat("Refs: Runge, J. (2018) 'Conditional independence testing based on a nearest-\n")
  cat("      neighbour estimator of conditional mutual information', AISTATS;\n")
  cat("      Zhang, K. et al. (2011) 'Kernel-based conditional independence test\n")
  cat("      and application in causal discovery', UAI.\n")
}
