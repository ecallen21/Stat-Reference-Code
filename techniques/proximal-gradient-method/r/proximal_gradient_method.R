# Proximal Gradient (Reference Sec 47.317)
# Native R via glmnet / celer via reticulate / RSpectra; Python via proxop / sklearn / from-scratch.
# Run with:  Rscript proximal_gradient_method.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  glmnet             -- LASSO / ridge / elastic-net via coord descent (very fast)\n")
  cat("  ncvreg             -- SCAD / MCP with proximal-flavour algorithms\n")
  cat("  admm.lasso         -- ADMM as an alternative to proximal gradient\n")
  cat("Python:\n")
  cat("  sklearn.linear_model.Lasso (coordinate descent)\n")
  cat("  celer (fast LASSO with proximal + safe screening)\n")
  cat("  proxop / pyproximal (general proximal operators)\n")
  cat("  From-scratch (see proximal_gradient_method.py)\n")
  cat("Refs: Combettes, P.L. & Wajs, V.R. (2005) 'Signal recovery by proximal\n")
  cat("      forward-backward splitting', SIAM MMS 4;  Parikh, N. & Boyd, S.\n")
  cat("      (2014) 'Proximal Algorithms', Foundations & Trends in Optim 1.\n")
}
