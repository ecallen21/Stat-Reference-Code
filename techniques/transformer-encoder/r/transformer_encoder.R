# Transformer encoder block (Reference §27.6)
# R via torch or keras3.
# Run with:  Rscript transformer_encoder.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  torch::nn_transformer_encoder_layer(d_model, nhead, dim_feedforward, activation, norm_first)\n")
  cat("  torch::nn_transformer_encoder(encoder_layer, num_layers)\n")
  cat("  keras3::layer_multi_head_attention + layer_normalization + custom feed-forward\n")
  cat("Positional encoding: sinusoidal (Vaswani 2017), learned, or rotary (RoPE).\n")
  cat("Python: torch.nn.TransformerEncoder, huggingface transformers.models.bert.BertEncoder,\n")
  cat("        flax.linen.SelfAttention + LayerNorm + Dense.\n")
}
