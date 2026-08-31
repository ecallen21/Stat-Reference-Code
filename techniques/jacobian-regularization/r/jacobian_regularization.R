# Jacobian regularization (Reference Ch 30 Robustness)
# R via reticulate + Python; analytic Jacobian is a few R lines for small nets.
# Run with:  Rscript jacobian_regularization.R

if (sys.nframe() == 0) {
  cat("R packages: analytic Jacobian is straightforward in base R.\n")
  cat("  numDeriv                     -- finite-difference Jacobian for any function\n")
  cat("  torch (R port)               -- torch::autograd_grad + manual Frobenius penalty\n")
  cat("Python:\n")
  cat("  torch.autograd.functional.jacobian + Frobenius penalty  (Hoffman 2019 reference)\n")
  cat("  jax.jacfwd, jax.jacrev       -- forward/reverse-mode Jacobians for functional APIs\n")
  cat("  jaxopt / flax                -- convenient training loops with Jacobian regularisation\n")
  cat("Refs: Hoffman, J., Roberts, D.A. & Yaida, S. (2019)\n")
  cat("      'Robust Learning with Jacobian Regularization', arXiv:1908.02729;\n")
  cat("      Jakubovitz, D. & Giryes, R. (2018)\n")
  cat("      'Improving DNN Robustness to Adversarial Attacks using Jacobian Regularization', ECCV.\n")
}
