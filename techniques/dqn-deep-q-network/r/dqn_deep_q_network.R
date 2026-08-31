# Deep Q-Network (Reference §28.3)
# R via torch or reticulate + Python (stable-baselines3, cleanrl).
# Run with:  Rscript dqn_deep_q_network.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  torch: manual DQN class -- nn_module Q, target Q, replay buffer, epsilon-greedy loop\n")
  cat("Python:\n")
  cat("  stable-baselines3.DQN         -- production PyTorch DQN\n")
  cat("  cleanrl/dqn.py                 -- single-file readable reference implementation\n")
  cat("  ray[rllib] DQN                 -- distributed / large-batch DQN\n")
  cat("Extensions (Rainbow, Hessel 2018):\n")
  cat("  * Double DQN — argmax and evaluation use different networks\n")
  cat("  * Duelling DQN — separate value and advantage streams\n")
  cat("  * Prioritised experience replay\n")
  cat("  * Multi-step returns / n-step Q-learning\n")
  cat("  * Categorical (C51 / Rainbow) — distributional RL\n")
  cat("  * Noisy Nets — parameter-space exploration\n")
  cat("Warning: DQN is off-policy and can diverge with function approximation\n")
  cat("  ('the deadly triad'); target networks + replay are the mitigations.\n")
}
