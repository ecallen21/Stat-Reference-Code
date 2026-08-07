# Dirichlet Process Gaussian Mixture (Reference §14.31)
# R via dirichletprocess or dpmixsim.
# Run with:  Rscript dirichlet_process_mixture.R

if (sys.nframe() == 0) {
  set.seed(0)
  y <- c(rnorm(60, -3, 0.6), rnorm(90, 0, 0.6), rnorm(50, 4, 0.6))
  if (requireNamespace("dirichletprocess", quietly = TRUE)) {
    cat("=== dirichletprocess::DirichletProcessGaussian ===\n")
    dp <- dirichletprocess::DirichletProcessGaussian(y)
    dp <- dirichletprocess::Fit(dp, its = 300, progressBar = FALSE)
    cat(sprintf("  final number of clusters: %d\n", length(unique(dp$clusterLabels))))
    print(table(dp$clusterLabels))
  }
}
