# PPO clipped-surrogate (Reference §28.5)
# R via torch or reticulate + Python.
# Run with:  Rscript ppo_clipped.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  torch: nn_module actor + critic; manual ratio + clip + K epochs update\n")
  cat("Python:\n")
  cat("  stable-baselines3.PPO         -- production PPO\n")
  cat("  cleanrl/ppo.py                 -- readable single-file reference\n")
  cat("  ray[rllib].PPOTrainer          -- distributed multi-worker PPO\n")
  cat("  TRL (huggingface):\n")
  cat("    * PPOTrainer for LLM fine-tuning with a reward model (RLHF)\n")
  cat("    * DPOTrainer, GRPOTrainer -- alternative preference-based fine-tuning\n")
  cat("Extensions:\n")
  cat("  * TRPO (Schulman 2015)  -- KL trust region via conjugate gradient\n")
  cat("  * GRPO (DeepSeek)        -- group-relative PPO for LLM reasoning\n")
  cat("  * DAPO / DPO / IPO       -- avoid RL entirely; direct preference optimisation\n")
}
