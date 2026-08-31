# Counterfactual fairness (Reference Ch 31 Fairness)
# R via reticulate + Python; native R has partial causal-fairness support.
# Run with:  Rscript counterfactual_fairness.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  dagitty                     -- DAG specification for SCMs\n")
  cat("  bnlearn                     -- Bayesian-network fitting for SCM\n")
  cat("  causalfair (community)      -- experimental counterfactual-fairness helpers\n")
  cat("Python:\n")
  cat("  doWhy                       -- SCM estimation + counterfactual queries\n")
  cat("  pyro / numpyro              -- probabilistic SCMs + posterior over U\n")
  cat("  aif360.algorithms.inprocessing.MetaFair (Celis 2019)\n")
  cat("Refs: Kusner, M.J., Loftus, J.R., Russell, C. & Silva, R. (2017)\n")
  cat("      'Counterfactual Fairness', NeurIPS.\n")
}
