# RLHF preferences: Bradley-Terry RM + DPO / PPO (Reference §28.10)
# R via reticulate + huggingface TRL.
# Run with:  Rscript rlhf_preferences.R

if (sys.nframe() == 0) {
  cat("R packages: no strong native R package for RLHF; use reticulate + Python.\n")
  cat("Python:\n")
  cat("  TRL (huggingface): PPOTrainer, DPOTrainer, GRPOTrainer, RLOOTrainer, ORPOTrainer\n")
  cat("  trlx (CarperAI): PPO / ILQL for text\n")
  cat("  OpenRLHF, TRLX-legacy, LLaMA-Factory, axolotl\n")
  cat("  d3rlpy for offline RL from preference logs\n")
  cat("Algorithms:\n")
  cat("  * SFT (supervised fine-tune) - warm start; BC over token sequences\n")
  cat("  * BT reward model (Christiano 2017) - MLE on pairwise preferences\n")
  cat("  * PPO with KL penalty to SFT model (InstructGPT/ChatGPT recipe)\n")
  cat("  * DPO (Rafailov 2023) - closed-form; skips explicit reward model\n")
  cat("  * IPO (Azar 2023) - identity-preference optimisation; alternative to DPO\n")
  cat("  * ORPO (Hong 2024) - odds-ratio preference; single-stage joint SFT+DPO\n")
  cat("  * GRPO (DeepSeek 2024) - group-relative advantage for reasoning training\n")
  cat("  * KTO (Ethayarajh 2024) - Kahneman-Tversky utility model; works without pairs\n")
  cat("  * RLHF-Free: Constitutional AI, RLAIF, self-play; various alignment recipes\n")
}
