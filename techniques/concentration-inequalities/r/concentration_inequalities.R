# Concentration inequalities (Reference Sec 46.14)
# From-scratch in both R and Python (analytic bounds); no dedicated package.
# Run with:  Rscript concentration_inequalities.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  No dedicated CRAN package -- the bounds are one-liners\n")
  cat("  Related: exp() * n * t^2 formulas; RSSampling stats helpers\n")
  cat("Python:\n")
  cat("  From-scratch numpy (see accompanying concentration_inequalities.py)\n")
  cat("  mlxtend.evaluate.mcnemar (McDiarmid-style tests)\n")
  cat("Refs: Boucheron, S., Lugosi, G. & Massart, P. (2013) Concentration\n")
  cat("      Inequalities: A Nonasymptotic Theory of Independence, OUP; Hoeffding\n")
  cat("      (1963) 'Probability inequalities for sums of bounded random\n")
  cat("      variables', JASA; Bernstein (1946) The Theory of Probabilities.\n")
}
