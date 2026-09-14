# Proximal Newton (Reference Sec 47.340)
# Native R via glmnet / ncvreg; Python via sklearn / from-scratch.
# Run with:  Rscript proximal_newton.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  glmnet                        -- Newton IRLS + coord descent for logistic-LASSO\n")
  cat("  ncvreg                        -- generalised for SCAD / MCP\n")
  cat("  biglasso                       -- large-scale variant\n")
  cat("Python:\n")
  cat("  sklearn.linear_model.LogisticRegression(penalty='l1', solver='saga')\n")
  cat("  celer.LogisticRegression       -- fast prox-Newton + safe screening\n")
  cat("  From-scratch (see proximal_newton.py)\n")
  cat("Refs: Lee, J.D., Sun, Y. & Saunders, M.A. (2014) 'Proximal Newton-type\n")
  cat("      methods for minimizing composite functions', SIAM J Optim 24;\n")
  cat("      Friedman, J., Hastie, T. & Tibshirani, R. (2010) 'Regularization\n")
  cat("      paths for generalized linear models via coordinate descent',\n")
  cat("      J Stat Softw 33(1) -- glmnet.\n")
}
