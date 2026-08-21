# Scaled dot-product / multi-head attention (Reference §27.5)
# R via torch or keras3.
# Run with:  Rscript attention_mechanism.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  torch::nn_multihead_attention(embed_dim, num_heads)\n")
  cat("  torch::nnf_scaled_dot_product_attention(q, k, v, is_causal=TRUE)   -- FlashAttention-style\n")
  cat("  keras3::layer_multi_head_attention(num_heads, key_dim)\n")
  cat("Python: torch.nn.MultiheadAttention, torch.nn.functional.scaled_dot_product_attention,\n")
  cat("        tensorflow.keras.layers.MultiHeadAttention, jax.nn.softmax + einsum.\n")
}
