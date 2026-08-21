# Transformer decoder block (Reference §27.x extra)
# R via torch.
# Run with:  Rscript transformer_decoder.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  torch::nn_transformer_decoder_layer(d_model, nhead, dim_feedforward, activation, norm_first)\n")
  cat("  torch::nn_transformer_decoder(decoder_layer, num_layers)\n")
  cat("Common decoder-only architectures (Python):\n")
  cat("  GPT / LLaMA / Mistral / DeepSeek / Qwen -- all pre-norm decoder-only causal LMs\n")
  cat("  T5 encoder-decoder — encoder + decoder with cross-attention\n")
  cat("  Whisper — encoder-decoder speech-to-text\n")
  cat("Standard tricks:\n")
  cat("  * RoPE (rotary positional encoding), ALiBi, GQA (grouped-query attention),\n")
  cat("    Flash-Attention, KV-cache for decoding.\n")
  cat("  * Speculative decoding, tree-of-thought, parallel sampling for inference speed.\n")
}
