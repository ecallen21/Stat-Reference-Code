# Elman RNN with BPTT (Reference §27.3)
# R via torch or keras3.
# Run with:  Rscript recurrent_nn.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  torch::nn_rnn(input_size, hidden_size, num_layers, batch_first=TRUE)\n")
  cat("  keras3::layer_simple_rnn(units, return_sequences=TRUE)\n")
  cat("  Bidirectional variants: torch::nn_rnn(bidirectional=TRUE), keras layer_bidirectional().\n")
  cat("  Use gradient clipping: torch::nn_utils_clip_grad_norm_(params, max_norm).\n")
  cat("Python: torch.nn.RNN, tensorflow.keras.layers.SimpleRNN, jax.nn.tanh + jax.lax.scan.\n")
}
