# DDPG + TD3 (Reference §28.x extra)
# R via reticulate + Python.
# Run with:  Rscript ddpg_td3.R

if (sys.nframe() == 0) {
  cat("R packages: no strong native R support; use reticulate + Python.\n")
  cat("Python:\n")
  cat("  stable-baselines3.DDPG / TD3            -- production off-policy continuous control\n")
  cat("  cleanrl/td3_continuous_action.py         -- readable reference implementation\n")
  cat("  ray[rllib].DDPGConfig                    -- distributed DDPG variants\n")
  cat("Extensions:\n")
  cat("  * DPG (Silver 2014), DDPG (Lillicrap 2015), TD3 (Fujimoto 2018)\n")
  cat("  * D4PG (Barth-Maron 2018)               -- distributional DDPG\n")
  cat("  * MPO / V-MPO (Abdolmaleki)             -- KL-regularised policy optimisation\n")
  cat("  * SAC (Haarnoja 2018)                    -- stochastic-policy alternative\n")
}
