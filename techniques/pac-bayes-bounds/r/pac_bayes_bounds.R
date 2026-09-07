# PAC-Bayes bounds (Reference Sec 46.15)
# From-scratch in both R and Python; no dedicated CRAN/PyPI package.
# Run with:  Rscript pac_bayes_bounds.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  No dedicated CRAN package -- one-line closed-form bounds\n")
  cat("  Related: mlr3verse learners (loss estimates for R side)\n")
  cat("Python:\n")
  cat("  From-scratch numpy (see accompanying pac_bayes_bounds.py)\n")
  cat("  pytorch-pacbayes / neurips code releases for deep-net variants\n")
  cat("Refs: McAllester, D. (1999) 'PAC-Bayesian model averaging', COLT;\n")
  cat("      Catoni, O. (2007) 'PAC-Bayesian supervised classification', IMS\n")
  cat("      Monograph; Seeger, M. (2002) 'PAC-Bayesian generalisation error\n")
  cat("      bounds for Gaussian process classification', JMLR.\n")
}
