# Integrated gradients (Reference Sec 47.30)
# Native R via lime + custom (no dedicated IG); Python via captum.
# Run with:  Rscript integrated_gradients.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  torch (Posit) + custom gradient path -- torch supports autograd IG\n")
  cat("  innsight (in dev) -- interpretability toolbox for neural networks in R\n")
  cat("  No dedicated CRAN IG package as of 2024\n")
  cat("Python:\n")
  cat("  captum.attr.IntegratedGradients (PyTorch, Meta AI)\n")
  cat("  alibi.explainers.IntegratedGradients (TensorFlow / Keras)\n")
  cat("  From-scratch numpy for differentiable f (see integrated_gradients.py)\n")
  cat("  shap (approximate IG variant)\n")
  cat("Refs: Sundararajan, M., Taly, A. & Yan, Q. (2017) 'Axiomatic attribution\n")
  cat("      for deep networks', ICML; Kokhlikyan, N. et al. (2020) Captum.\n")
}
