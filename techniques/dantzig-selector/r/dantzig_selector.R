# Dantzig selector (Reference Sec 6.18)
# Native R via flare / hdi; Python via scipy.optimize.linprog / cvxpy.
# Run with:  Rscript dantzig_selector.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  flare::slim(method='dantzig')  -- fast LP-based Dantzig selector\n")
  cat("  hdi::lasso.proj                 -- variant with post-selection inference\n")
  cat("  glmnet + lasso as related alternative\n")
  cat("Python:\n")
  cat("  cvxpy (concise LP form)          -- rapid prototyping\n")
  cat("  scipy.optimize.linprog + custom  -- see dantzig_selector.py\n")
  cat("  skglm / celer (fast solvers)\n")
  cat("Refs: Candes, E. & Tao, T. (2007) 'The Dantzig selector: statistical\n")
  cat("      estimation when p is much larger than n', Ann Stat 35(6): 2313-\n")
  cat("      2351; Bickel, Ritov & Tsybakov (2009) 'Simultaneous analysis of\n")
  cat("      Lasso and Dantzig selector', Ann Stat.\n")
}
