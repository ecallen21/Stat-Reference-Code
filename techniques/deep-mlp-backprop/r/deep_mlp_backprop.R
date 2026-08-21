# Deep MLP with back-propagation (Reference §27.1)
# R via torch, keras, or nnet (single-hidden-layer only).
# Run with:  Rscript deep_mlp_backprop.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  torch::nn_module + nn_linear + nn_relu             -- LibTorch bindings; full flexibility\n")
  cat("  keras / keras3 (Keras via reticulate)              -- TF/Keras from R\n")
  cat("  nnet::nnet(y ~ ., size=k)                          -- classical 1-hidden-layer MLP\n")
  cat("  RSNNS::mlp                                          -- Stuttgart Neural Network Simulator\n")
  cat("Python: torch.nn.Sequential(Linear, ReLU, ..., Linear), sklearn.neural_network.MLPClassifier,\n")
  cat("        tensorflow.keras.Sequential.\n")
}
