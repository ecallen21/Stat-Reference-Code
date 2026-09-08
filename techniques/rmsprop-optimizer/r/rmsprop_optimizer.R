# RMSprop (Reference Sec 47.106)
# Native R via torch; Python via torch.optim / keras.
# Run with:  Rscript rmsprop_optimizer.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  torch::optim_rmsprop     -- torch R bindings\n")
  cat("  keras::optimizer_rmsprop -- keras R bindings\n")
  cat("Python:\n")
  cat("  torch.optim.RMSprop\n")
  cat("  keras.optimizers.RMSprop\n")
  cat("  jax.optax.rmsprop\n")
  cat("  from-scratch             -- see rmsprop_optimizer.py\n")
  cat("Refs: Tieleman & Hinton (2012) Coursera lecture 6e; Kingma & Ba (2015)\n")
  cat("      'Adam: A method for stochastic optimization', ICLR.\n")
}
