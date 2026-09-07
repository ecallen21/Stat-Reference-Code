# Grouped-query attention (GQA) (Reference Sec 47.37)
# LLM architecture element; Python via HF transformers Llama / Mistral / Gemma modules.
# Run with:  Rscript grouped_query_attention.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  No R implementations for GQA\n")
  cat("Python:\n")
  cat("  transformers (HF) LlamaAttention / MistralAttention modules use GQA\n")
  cat("  vLLM / flashattention -- optimised GQA kernels\n")
  cat("  torch: implement Q/K/V heads with n_kv_heads < n_q_heads and expand\n")
  cat("Refs: Ainslie, J. et al. (2023) 'GQA: training generalized multi-query\n")
  cat("      transformer models from multi-head checkpoints', EMNLP; Shazeer, N.\n")
  cat("      (2019) 'Fast transformer decoding: one write-head is all you need'\n")
  cat("      (MQA precursor).\n")
}
