# Matrix completion via SVT (Reference Sec 47.43)
# Native R via softImpute; Python via fancyimpute / from-scratch.
# Run with:  Rscript matrix_completion_svt.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  softImpute::softImpute       -- ALS-based nuclear-norm completion\n")
  cat("  filling                      -- SVT, OptSpace, ADMM in one API\n")
  cat("  ROpenCVLite / imager         -- image-inpainting demos\n")
  cat("Python:\n")
  cat("  fancyimpute.SoftImpute       -- direct port of softImpute\n")
  cat("  scipy.sparse.linalg.svds     -- large-scale sparse SVD in SVT loops\n")
  cat("  from-scratch                 -- see matrix_completion_svt.py\n")
  cat("Refs: Cai, Candes & Shen (2010) SIAM J Optim 20(4); Candes & Recht (2009)\n")
  cat("      FoCM 9; Mazumder, Hastie & Tibshirani (2010) JMLR 11.\n")
}
