# Evidential deep learning (Reference Ch 29 Uncertainty Quantification)
# R via reticulate + Python; native R support is very limited.
# Run with:  Rscript evidential_deep_learning.R

if (sys.nframe() == 0) {
  cat("R packages: no first-class R support; use reticulate + Python.\n")
  cat("Python:\n")
  cat("  evidential-deep-learning-pytorch   -- Sensoy 2018 reference implementation\n")
  cat("  edl-pytorch                        -- classification Dirichlet head + evidential loss\n")
  cat("  evidential_regression              -- Amini 2020 Normal-Inverse-Gamma head\n")
  cat("  gpytorch                           -- higher-order posteriors via GPs (comparison)\n")
  cat("Refs: Sensoy, M., Kaplan, L. & Kandemir, M. (2018) 'Evidential Deep Learning\n")
  cat("      to Quantify Classification Uncertainty', NeurIPS;\n")
  cat("      Amini, A., Schwarting, W., Soleimany, A. & Rus, D. (2020)\n")
  cat("      'Deep Evidential Regression', NeurIPS.\n")
}
