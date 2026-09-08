# Support vector regression (SVR) (Reference Sec 5.17)
# Native R via e1071 / kernlab; Python via scikit-learn.
# Run with:  Rscript support_vector_regression.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  e1071::svm(type = 'eps-regression')  -- libsvm wrapper\n")
  cat("  kernlab::ksvm(type = 'eps-svr')       -- kernlab alternative\n")
  cat("  liquidSVM                             -- fast SVM/SVR C++ core\n")
  cat("Python:\n")
  cat("  sklearn.svm.SVR                       -- libsvm-based, RBF / linear / poly\n")
  cat("  sklearn.svm.LinearSVR                 -- LIBLINEAR-based, large n\n")
  cat("  cvxopt / qpsolvers for from-scratch dual QP\n")
  cat("Refs: Drucker, H. et al. (1996) 'Support vector regression machines',\n")
  cat("      NIPS 9; Smola, A.J. & Scholkopf, B. (2004) 'A tutorial on support\n")
  cat("      vector regression', Stat Comp 14(3): 199-222.\n")
}
