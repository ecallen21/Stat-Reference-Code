# Autoencoder + denoising autoencoder (Reference §27.7)
# R via torch or keras3.
# Run with:  Rscript autoencoder.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  torch::nn_sequential(nn_linear(d, h), nn_relu(), nn_linear(h, k))  -- encoder\n")
  cat("  keras3 encoder + decoder Sequential; compile(loss='mse')\n")
  cat("  ANN2::autoencoder, h2o::h2o.deeplearning(autoencoder=TRUE)\n")
  cat("Python: torch.nn.Sequential, tensorflow.keras.Model with two Sequential branches,\n")
  cat("        sklearn.preprocessing MinMaxScaler + torch autoencoder training loop.\n")
}
