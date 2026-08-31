# Generalised Advantage Estimation (Reference §28.12)
# R via torch or reticulate + Python.
# Run with:  Rscript gae_advantage_estimation.R

if (sys.nframe() == 0) {
  cat("R packages: no strong native R implementation; use reticulate + Python.\n")
  cat("Python:\n")
  cat("  stable-baselines3.common.buffers.RolloutBuffer.compute_returns_and_advantages\n")
  cat("  cleanrl/ppo.py -- readable single-file GAE implementation\n")
  cat("  ray[rllib] AlgorithmConfig(gae_lambda=...) -- configurable per-algo\n")
  cat("Where used:\n")
  cat("  * PPO (Schulman 2017)  -- the standard companion; lambda ~ 0.95\n")
  cat("  * A2C / A3C / TRPO      -- same recipe\n")
  cat("  * IMPALA (Espeholt 2018) -- V-trace, an off-policy correction of GAE\n")
  cat("  * SAC / DDPG / TD3       -- do NOT use GAE (off-policy TD-based)\n")
}
