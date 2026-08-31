# Out-of-distribution detection (Reference Ch 29 Uncertainty Quantification)
# R via reticulate + Python; some native R options for classical methods.
# Run with:  Rscript ood_detection.R

if (sys.nframe() == 0) {
  cat("R packages (classical):\n")
  cat("  mvoutlier                   -- multivariate outlier + Mahalanobis-style detection\n")
  cat("  isotree                     -- isolation forest for tabular OOD\n")
  cat("  robustbase::covMcd          -- robust Mahalanobis (Minimum Covariance Determinant)\n")
  cat("Python (deep-learning OOD):\n")
  cat("  pytorch-ood                 -- MSP, Energy, Mahalanobis, ODIN, ViM, etc.\n")
  cat("  cleanlab                    -- CL-based OOD flagging\n")
  cat("  torchdrift                  -- covariate + label-shift detection\n")
  cat("  openood                     -- unified OOD benchmark suite\n")
  cat("Refs: Hendrycks & Gimpel (2017) 'A Baseline for Detecting Misclassified\n")
  cat("      and OOD Examples in NNs (MSP)', ICLR; Liu et al. (2020) 'Energy-based\n")
  cat("      Out-of-Distribution Detection', NeurIPS; Lee et al. (2018)\n")
  cat("      'A Simple Unified Framework for Detecting OOD Samples (Mahalanobis)', NeurIPS.\n")
}
