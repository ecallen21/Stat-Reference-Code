# LoRA -- parameter-efficient fine-tuning (Reference Sec 47.24)
# Deep-learning training technique; Python via HF peft / trl.
# Run with:  Rscript lora_peft.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  No R implementations for LoRA fine-tuning\n")
  cat("Python:\n")
  cat("  peft (HuggingFace) -- LoRA / QLoRA / DoRA / AdaLoRA / IA3 / prefix\n")
  cat("  trl (HuggingFace) -- SFTTrainer with peft integration\n")
  cat("  bitsandbytes -- 4-bit / 8-bit quantised base weights for QLoRA\n")
  cat("  unsloth -- optimised LoRA training kernels\n")
  cat("  lit-gpt / axolotl -- higher-level fine-tuning frameworks\n")
  cat("Refs: Hu, E.J. et al. (2021) 'LoRA: Low-Rank Adaptation of Large\n")
  cat("      Language Models', ICLR 2022; Dettmers, T. et al. (2023) 'QLoRA:\n")
  cat("      efficient finetuning of quantized LLMs', NeurIPS; Liu, S.-Y. et al.\n")
  cat("      (2024) 'DoRA: Weight-Decomposed Low-Rank Adaptation', ICML.\n")
}
