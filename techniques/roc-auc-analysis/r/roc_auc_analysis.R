# ROC / AUC (Reference §21.5)
# R via pROC::roc (with DeLong CI) or ROCR.
# Run with:  Rscript roc_auc_analysis.R

if (sys.nframe() == 0) {
  set.seed(0); n_pos <- 100; n_neg <- 200
  y <- c(rep(1, n_pos), rep(0, n_neg))
  scores <- c(rnorm(n_pos, 1, 1), rnorm(n_neg, 0, 1))
  if (requireNamespace("pROC", quietly = TRUE)) {
    cat("=== pROC::roc ===\n")
    r <- pROC::roc(y, scores)
    print(r)
    cat(sprintf("  DeLong 95%% CI: (%.4f, %.4f)\n",
                pROC::ci.auc(r, method = "delong")[1],
                pROC::ci.auc(r, method = "delong")[3]))
  }
}
