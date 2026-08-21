# Reinforcement learning basics (Reference §27.x extra)
# R via ReinforcementLearning package or reticulate + Python gymnasium + SB3.
# Run with:  Rscript reinforcement_learning_basics.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  ReinforcementLearning::ReinforcementLearning(state, action, reward, ...)\n")
  cat("  MDPtoolbox::mdp_Q_learning / mdp_policy_iteration / mdp_value_iteration\n")
  cat("  reticulate + gymnasium + stable-baselines3 (PPO, DQN, SAC, TD3, A2C)\n")
  cat("  reticulate + torch + ray[rllib]\n")
  cat("Common algorithms:\n")
  cat("  * Value: Q-learning, SARSA, DQN, Rainbow, C51, IQN\n")
  cat("  * Policy: REINFORCE, A2C/A3C, PPO, TRPO, SAC (continuous), TD3\n")
  cat("  * Model-based: Dyna-Q, MuZero, MBPO\n")
  cat("  * Offline RL: BCQ, CQL, IQL; batch-RL for medical / finance\n")
  cat("  * RLHF: PPO on preference-model reward for LLM alignment (InstructGPT).\n")
}
