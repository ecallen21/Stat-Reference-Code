# LSTM and GRU (Reference §27.4)
# R via torch or keras3.
# Run with:  Rscript lstm_gru.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  torch::nn_lstm(input_size, hidden_size, num_layers, batch_first=TRUE)\n")
  cat("  torch::nn_gru(input_size, hidden_size, num_layers, batch_first=TRUE)\n")
  cat("  keras3::layer_lstm(units) / layer_gru(units) / layer_bidirectional()\n")
  cat("  Set forget_bias=1 (LSTM) via manual init for a small performance boost.\n")
  cat("Python: torch.nn.LSTM / GRU, tensorflow.keras.layers.LSTM / GRU / Bidirectional,\n")
  cat("        jax.experimental.stax.LSTM / GRU.\n")
}
