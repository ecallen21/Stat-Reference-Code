# VC (Vapnik-Chervonenkis) dimension (Reference Sec 46.11)
# No dedicated CRAN/PyPI package; from-scratch numpy in Python.
# Run with:  Rscript vc_dimension.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  No dedicated CRAN package -- implement from scratch\n")
  cat("  Related: e1071 (SVM, empirical VC estimation via SVM radius/margin)\n")
  cat("  Related: mlr3verse learning-curve utilities\n")
  cat("Python:\n")
  cat("  From-scratch numpy (see accompanying vc_dimension.py)\n")
  cat("  mlxtend.evaluate.vc_dim (empirical estimator via Vapnik's algorithm)\n")
  cat("Refs: Vapnik, V.N. & Chervonenkis, A.Ya. (1971) 'On the uniform\n")
  cat("      convergence of relative frequencies of events to their\n")
  cat("      probabilities', Theory Probab Appl 16(2); Vapnik (1998)\n")
  cat("      Statistical Learning Theory, Wiley.\n")
}
