# Neural tangent kernel (NTK) (Reference Sec 46.19)
# Theory / deep-learning; Python via neural-tangents (Google).
# Run with:  Rscript neural_tangent_kernel.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  No R implementations for NTK / infinite-width NN theory\n")
  cat("Python:\n")
  cat("  neural-tangents (Google) -- infinite-width NN + NTK computation in JAX\n")
  cat("  ntk-pytorch / functorch -- finite-width NTK estimation\n")
  cat("  jax.grad + linear algebra -- from-scratch NTK for shallow nets\n")
  cat("Refs: Jacot, A., Gabriel, F. & Hongler, C. (2018) 'Neural tangent kernel:\n")
  cat("      convergence and generalization in neural networks', NeurIPS; Lee,\n")
  cat("      J. et al. (2019) 'Wide neural networks of any depth evolve as\n")
  cat("      linear models under gradient descent', NeurIPS.\n")
}
