# Meta-analysis (Reference §20.1)
# R via metafor::rma (Viechtbauer).
# Run with:  Rscript meta_analysis.R

if (sys.nframe() == 0) {
  set.seed(0); k <- 8
  theta <- rnorm(k, 0.35, 0.15)
  vi <- runif(k, 0.03, 0.12)
  yi <- rnorm(k, theta, sqrt(vi))
  if (requireNamespace("metafor", quietly = TRUE)) {
    cat("=== metafor::rma (DerSimonian-Laird) ===\n")
    print(metafor::rma(yi = yi, vi = vi, method = "DL"))
  }
}
