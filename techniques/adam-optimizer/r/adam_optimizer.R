# Adam / SGD / RMSprop / AdamW (Reference §27.11)
# R via torch.
# Run with:  Rscript adam_optimizer.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  torch::optim_sgd(params, lr, momentum, nesterov)\n")
  cat("  torch::optim_rmsprop(params, lr, alpha)\n")
  cat("  torch::optim_adam(params, lr, betas=c(0.9, 0.999), eps=1e-8, weight_decay)\n")
  cat("  torch::optim_adamw(params, lr, weight_decay)   -- decoupled weight decay\n")
  cat("  torch::optim_lbfgs                              -- L-BFGS for small problems\n")
  cat("Python: torch.optim.SGD / Adam / AdamW / RMSprop / LBFGS,\n")
  cat("        jax.optimizers / optax (adamw, lamb, adafactor, lion, ...).\n")
  cat("Schedulers: warmup + cosine / one-cycle / step / plateau — combine with any optimiser.\n")
}
