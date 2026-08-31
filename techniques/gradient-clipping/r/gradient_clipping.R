# Gradient clipping (Reference Ch 30 Robustness)
# R via reticulate + Python; native torch (R) exposes clip_grad_norm.
# Run with:  Rscript gradient_clipping.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  torch (R port)              -- torch::nn_utils_clip_grad_norm_ / _value_\n")
  cat("  keras3 / tensorflow (R)     -- optimizer_adam(clipnorm = ...), clipvalue = ...\n")
  cat("Python:\n")
  cat("  torch.nn.utils.clip_grad_norm_ / clip_grad_value_\n")
  cat("  tf.clip_by_global_norm / tf.clip_by_value\n")
  cat("  jax.example_libraries.optimizers.clip_grads\n")
  cat("Refs: Pascanu, R., Mikolov, T. & Bengio, Y. (2013)\n")
  cat("      'On the difficulty of training recurrent neural networks', ICML.\n")
}
