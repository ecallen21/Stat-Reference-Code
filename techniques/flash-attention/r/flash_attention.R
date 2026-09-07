# FlashAttention (Reference Sec 47.38)
# CUDA-level attention kernel; Python via flash-attn / xFormers / torch SDPA.
# Run with:  Rscript flash_attention.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  No R implementations for CUDA kernels\n")
  cat("Python:\n")
  cat("  flash-attn (Dao) -- reference CUDA kernel + Python bindings\n")
  cat("  xFormers memory_efficient_attention -- Meta AI kernel family\n")
  cat("  torch.nn.functional.scaled_dot_product_attention -- selects FA/CUDA/SDPA\n")
  cat("  vLLM PagedAttention + FlashAttention -- inference-server integration\n")
  cat("Refs: Dao, T., Fu, D.Y., Ermon, S., Rudra, A. & Re, C. (2022)\n")
  cat("      'FlashAttention: fast and memory-efficient exact attention with\n")
  cat("      IO-awareness', NeurIPS; Dao, T. (2023) 'FlashAttention-2', ICLR.\n")
}
