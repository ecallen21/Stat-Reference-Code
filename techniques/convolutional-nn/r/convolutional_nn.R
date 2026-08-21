# Convolutional Neural Network (Reference §27.2)
# R via torch or keras3.
# Run with:  Rscript convolutional_nn.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  torch::nn_conv2d(in_ch, out_ch, kernel_size)   +   nn_max_pool2d + nn_relu\n")
  cat("  keras3::layer_conv_2d + layer_max_pooling_2d + layer_flatten + layer_dense\n")
  cat("  rTensor / OpenImageR — lightweight image / conv utilities\n")
  cat("  For explanation: gradient-based (integrated gradients, Grad-CAM) via captum in Python.\n")
  cat("Python: torch.nn.Conv2d, tensorflow.keras.layers.Conv2D, jax.lax.conv_general_dilated.\n")
}
