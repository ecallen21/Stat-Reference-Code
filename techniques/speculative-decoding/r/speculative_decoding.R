# Speculative decoding (Reference Sec 47.21)
# Inference-time acceleration; no R implementations. Python via vLLM.
# Run with:  Rscript speculative_decoding.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  No R implementations for LLM inference internals.\n")
  cat("Python:\n")
  cat("  vLLM      -- production speculative decoding + PagedAttention\n")
  cat("  Hugging Face transformers >= 4.35: assisted_generation\n")
  cat("  TGI (Text Generation Inference, HF)\n")
  cat("  Medusa (parallel draft heads on the target model)\n")
  cat("  EAGLE / Lookahead (variant draft strategies)\n")
  cat("Refs: Leviathan, Y., Kalman, M. & Matias, Y. (2023) 'Fast inference from\n")
  cat("      transformers via speculative decoding', ICML; Chen, C. et al.\n")
  cat("      (2023) 'Accelerating large language model decoding with speculative\n")
  cat("      sampling', arXiv:2302.01318.\n")
}
