# Residual connections (Reference §27.x extra)
# R via torch or keras3.
# Run with:  Rscript residual_connections.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  torch::nn_module with y = x + self$block(x)   -- explicit residual class\n")
  cat("  keras3::layer_add(list(x, block(x)))          -- functional API residual\n")
  cat("Standard architectures with residuals:\n")
  cat("  ResNet-50/101/152 (He et al. 2016)  -- CNN backbone; identity + conv-BN-relu block\n")
  cat("  Transformer encoder / decoder blocks -- residual around attention + FFN sublayers\n")
  cat("  DenseNet  -- concatenative skip connections (Huang et al. 2017)\n")
  cat("  Highway networks (Srivastava et al. 2015)  -- gated residual\n")
  cat("  Deep-Norm / ReZero / LayerScale — init tricks that scale residual contribution.\n")
}
