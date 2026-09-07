# DPO -- Direct Preference Optimization (Reference Sec 47.25)
# Deep-LM alignment; Python via HuggingFace trl.
# Run with:  Rscript dpo_direct_preference_optimization.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  No R implementations for DPO (LLM fine-tuning technique)\n")
  cat("Python:\n")
  cat("  trl.DPOTrainer / DPOConfig  -- Hugging Face TRL reference impl\n")
  cat("  trl.IPOTrainer / KTOTrainer -- Rafailov+Ethayarajh variants\n")
  cat("  transformers + PEFT + trl   -- full LoRA + DPO stack\n")
  cat("  axolotl / lit-gpt           -- higher-level DPO pipelines\n")
  cat("Refs: Rafailov, R. et al. (2023) 'Direct Preference Optimization: your\n")
  cat("      language model is secretly a reward model', NeurIPS; Azar, M.G.\n")
  cat("      et al. (2023) 'A general theoretical paradigm to understand\n")
  cat("      learning from human preferences' (IPO); Ethayarajh, K. et al.\n")
  cat("      (2024) 'KTO: Model alignment as prospect theoretic optimization'.\n")
}
