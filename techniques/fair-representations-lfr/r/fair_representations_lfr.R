# Learning Fair Representations / INLP-style projection (Reference Ch 31 Fairness)
# R via reticulate + Python; native R fine for the linear-projection variant.
# Run with:  Rscript fair_representations_lfr.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  fairml                      -- linear-projection fair pre-processors\n")
  cat("  fairness / fairmodels       -- fairness pipelines that accept debiased data\n")
  cat("Python:\n")
  cat("  aif360.algorithms.preprocessing.LFR      (Zemel 2013 prototype variant)\n")
  cat("  concept-erasure                          (Belrose 2023, generalisation of INLP)\n")
  cat("  inlp-oracle (Ravfogel 2020 reference implementation)\n")
  cat("Refs: Zemel, R., Wu, Y., Swersky, K., Pitassi, T. & Dwork, C. (2013)\n")
  cat("      'Learning Fair Representations', ICML;\n")
  cat("      Ravfogel, S. et al. (2020) 'Null It Out: Guarding Protected Attributes\n")
  cat("      by Iterative Nullspace Projection (INLP)', ACL.\n")
}
