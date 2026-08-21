# Mixture of Experts (Reference §27.x extra)
# R via torch or reticulate + Python (fairscale, DeepSpeed, MegaBlocks).
# Run with:  Rscript mixture_of_experts.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  torch: manual MoE layer (gate + top-k mask + per-expert MLP)\n")
  cat("  reticulate + Python:\n")
  cat("    fairscale.nn.MOELayer (FBAI)\n")
  cat("    DeepSpeed-MoE (Microsoft) — token routing at scale\n")
  cat("    MegaBlocks (Databricks) — sparse-matmul MoE for GPU efficiency\n")
  cat("    tutel (Microsoft) — MoE routing kernels for multi-GPU\n")
  cat("Production MoE models:\n")
  cat("  * Switch Transformer (Fedus 2022)  -- top-1 routing; ~1 trillion params\n")
  cat("  * GLaM, GShard, ST-MoE               -- top-2 routing\n")
  cat("  * Mixtral 8x7B, DeepSeek-V2, Qwen2-MoE, Grok-1 -- open-weight MoE LLMs\n")
  cat("  * Vision MoE (V-MoE, Ridnik 2022)   -- ViT + MoE\n")
  cat("Standard tricks: capacity factor, expert-choice routing (Zhou 2022),\n")
  cat("                  auxiliary-loss-free routing (DeepSeek 2024).\n")
}
