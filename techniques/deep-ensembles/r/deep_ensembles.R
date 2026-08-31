# Deep ensembles (Reference Ch 29 Uncertainty Quantification)
# R via reticulate + Python (torch / tensorflow) — no canonical R package.
# Run with:  Rscript deep_ensembles.R

if (sys.nframe() == 0) {
  cat("R packages: no canonical R package; use reticulate + Python.\n")
  cat("Python:\n")
  cat("  torch nn.Module           -- train K MLPs with Gaussian-NLL head, different inits\n")
  cat("  tensorflow-probability     -- tfp.layers.DenseVariational for reference\n")
  cat("  keras Ensemble            -- manual K-fold model averaging\n")
  cat("  uncertainty-toolbox       -- calibration + sharpness metrics for ensemble output\n")
  cat("R alternatives:\n")
  cat("  torch (R port)            -- torch::nn_module + explicit ensemble loop\n")
  cat("  brulee, tabnet            -- tidymodels neural fits; ensemble via workflowsets\n")
  cat("Refs: Lakshminarayanan et al. (2017) 'Simple and Scalable Predictive Uncertainty\n")
  cat("      Estimation using Deep Ensembles', NeurIPS.\n")
}
