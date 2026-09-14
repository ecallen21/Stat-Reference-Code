# Straight-Through Estimator (Reference Sec 47.316)
# Native R via torch (R) + custom autograd; Python via torch.autograd.Function / from-scratch.
# Run with:  Rscript straight_through_estimator.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  torch (R) with custom autograd_function\n")
  cat("  reticulate + torch\n")
  cat("Python:\n")
  cat("  torch.autograd.Function.apply with custom .backward = identity\n")
  cat("  torch.ao.quantization (INT8 with STE)\n")
  cat("  vector_quantize_pytorch (VQ-VAE STE)\n")
  cat("  From-scratch (see straight_through_estimator.py)\n")
  cat("Refs: Bengio, Y., Leonard, N. & Courville, A. (2013) 'Estimating or\n")
  cat("      propagating gradients through stochastic neurons for conditional\n")
  cat("      computation', arXiv:1308.3432.\n")
}
