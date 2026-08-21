# Dropout + BatchNorm (Reference §27.10)
# R via torch or keras3.
# Run with:  Rscript dropout_batchnorm.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  torch::nn_dropout(p=0.5)  torch::nn_batch_norm1d(features) / nn_batch_norm2d\n")
  cat("  keras3::layer_dropout(rate=0.5)  keras3::layer_batch_normalization()\n")
  cat("Variants:\n")
  cat("  LayerNorm (across features per example)   -- transformers; use torch::nn_layer_norm\n")
  cat("  GroupNorm (across group of channels)       -- helpful for small batches; nn_group_norm\n")
  cat("  RMSNorm (variance-only)                    -- modern LLM default; simpler than LayerNorm\n")
  cat("  Weight decay (L2) — orthogonal regulariser that pairs well with either.\n")
}
