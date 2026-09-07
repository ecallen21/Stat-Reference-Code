# Youden's J / ROC01 optimal cutpoint (Reference Sec 26.10)
# Native R via cutpointr / OptimalCutpoints; Python via sklearn / scipy.
# Run with:  Rscript youden_optimal_cutpoint.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  cutpointr                 -- Youden / ROC01 / max acc / cost-weighted\n")
  cat("  OptimalCutpoints          -- >20 optimality criteria + bootstrap CIs\n")
  cat("  pROC::coords              -- Youden and other cut-offs on roc objects\n")
  cat("Python:\n")
  cat("  sklearn.metrics.roc_curve + custom argmax of TPR - FPR\n")
  cat("  From-scratch numpy (see youden_optimal_cutpoint.py)\n")
  cat("Refs: Youden, W.J. (1950) 'Index for rating diagnostic tests', Cancer 3;\n")
  cat("      Perkins & Schisterman (2006) 'The inconsistency of `optimal'\n")
  cat("      cut-points', Am J Epi 163(7).\n")
}
