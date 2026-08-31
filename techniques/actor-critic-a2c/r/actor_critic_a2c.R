# Actor-critic A2C (Reference §28.4)
# R via torch or reticulate + Python (stable-baselines3, cleanrl).
# Run with:  Rscript actor_critic_a2c.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  torch: nn_module actor (softmax head) + critic (value head); manual TD loop\n")
  cat("Python:\n")
  cat("  stable-baselines3.A2C                                -- production A2C / A3C\n")
  cat("  stable-baselines3.PPO / SAC / DDPG / TD3            -- alternative on-policy / off-policy\n")
  cat("  cleanrl/a2c_continuous_action.py                     -- single-file reference\n")
  cat("  ray[rllib].A2CTrainer, ray[rllib].APPOTrainer        -- distributed A2C\n")
  cat("Extensions:\n")
  cat("  * A3C (Mnih 2016) -- asynchronous, multi-worker\n")
  cat("  * GAE (Schulman 2016) -- lambda-averaged advantage; see gae-advantage-estimation\n")
  cat("  * IMPALA (Espeholt 2018) -- distributed importance-weighted actor-critic\n")
}
