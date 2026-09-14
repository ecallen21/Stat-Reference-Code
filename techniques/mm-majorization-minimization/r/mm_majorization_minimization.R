# Majorization-Minimization (Reference Sec 47.322)
# Native R via base R custom loops; Python via scipy / from-scratch.
# Run with:  Rscript mm_majorization_minimization.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  glmnet / ncvreg              -- MM/coord descent for LASSO / SCAD\n")
  cat("  Rcpp custom implementations  -- easy to code from scratch\n")
  cat("  lava / lme4                  -- IRLS is an MM instance\n")
  cat("Python:\n")
  cat("  sklearn.linear_model / statsmodels IRLS routines (MM internals)\n")
  cat("  cvxpy for a declarative surrogate approach\n")
  cat("  From-scratch (see mm_majorization_minimization.py)\n")
  cat("Refs: Hunter, D.R. & Lange, K. (2004) 'A tutorial on MM algorithms',\n")
  cat("      Am Stat 58(1);  Lange, K. (2016) 'MM Optimization Algorithms',\n")
  cat("      SIAM.\n")
}
