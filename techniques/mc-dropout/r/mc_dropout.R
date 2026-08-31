# MC dropout (Reference Ch 29 Uncertainty Quantification)
# R via reticulate + Python (torch / tensorflow) — no canonical R package.
# Run with:  Rscript mc_dropout.R

if (sys.nframe() == 0) {
  cat("R packages: no canonical R package; use reticulate + Python.\n")
  cat("Python:\n")
  cat("  torch nn.Dropout          -- keep model.train() at inference for MC samples\n")
  cat("  tensorflow-probability    -- tfp.layers.DenseVariational + dropout equivalence\n")
  cat("  keras Dropout             -- training=True at inference for MC prediction\n")
  cat("  uncertainty-toolbox       -- calibration metrics for MC-dropout outputs\n")
  cat("R alternatives:\n")
  cat("  torch (R port)            -- torch::nn_dropout kept active at inference via mode='train'\n")
  cat("Refs: Gal, Y. & Ghahramani, Z. (2016) 'Dropout as a Bayesian Approximation:\n")
  cat("      Representing Model Uncertainty in Deep Learning', ICML.\n")
}
