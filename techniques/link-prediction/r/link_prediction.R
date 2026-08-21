# Link prediction (Reference §24.7)
# R via igraph + linkprediction.
# Run with:  Rscript link_prediction.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  igraph::similarity(g, method='jaccard' | 'dice' | 'invlogweighted')\n")
  cat("  linkprediction::proxfun(g, method='aa' | 'cn' | 'ra' | 'pa' | 'katz')\n")
  cat("  linkprediction::proxfun(g, method='rwr', alpha=0.85)   -- random-walk with restart\n")
  cat("  pROC::roc(labels, scores)                              -- AUC evaluation\n")
}
