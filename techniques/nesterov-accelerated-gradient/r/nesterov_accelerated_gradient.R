# Nesterov Accelerated Gradient (Reference Sec 47.107)
# Native R via torch; Python via torch / jax.
# Run with:  Rscript nesterov_accelerated_gradient.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  torch::optim_sgd(momentum, nesterov=TRUE)\n")
  cat("  glmnet                        -- coord descent with warm starts (analogous rate)\n")
  cat("Python:\n")
  cat("  torch.optim.SGD(momentum, nesterov=True)\n")
  cat("  keras.optimizers.SGD(nesterov=True)\n")
  cat("  jax.optax.sgd + nesterov=True\n")
  cat("  fista / proxop            -- proximal-gradient FISTA implementations\n")
  cat("  from-scratch              -- see nesterov_accelerated_gradient.py\n")
  cat("Refs: Nesterov (1983) Dokl Akad Nauk SSSR 269; Beck & Teboulle (2009)\n")
  cat("      'FISTA', SIAM J Imaging Sci 2(1); O'Donoghue & Candes (2015)\n")
  cat("      'Adaptive restart for accelerated gradient schemes', FoCM 15.\n")
}
