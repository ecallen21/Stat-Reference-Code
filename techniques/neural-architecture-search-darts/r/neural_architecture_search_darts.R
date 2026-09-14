# DARTS - Differentiable Architecture Search (Reference Sec 47.279)
# Native R via reticulate + PyTorch DARTS; Python via NNI / AutoKeras / from-scratch.
# Run with:  Rscript neural_architecture_search_darts.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  reticulate + PyTorch DARTS / NNI (Python-native)\n")
  cat("  torch (R) with custom bilevel optimisation\n")
  cat("Python:\n")
  cat("  quark0/darts (PyTorch reference implementation)\n")
  cat("  Microsoft NNI NAS module\n")
  cat("  AutoKeras (higher-level NAS wrapper)\n")
  cat("  From-scratch (see neural_architecture_search_darts.py)\n")
  cat("Refs: Liu, H., Simonyan, K. & Yang, Y. (2019) 'DARTS: Differentiable\n")
  cat("      architecture search', ICLR.\n")
}
