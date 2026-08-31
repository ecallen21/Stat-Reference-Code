# Soft Actor-Critic (Reference §28.x extra)
# R via reticulate + Python.
# Run with:  Rscript sac_soft_actor_critic.R

if (sys.nframe() == 0) {
  cat("R packages: no strong native R support; use reticulate + Python.\n")
  cat("Python:\n")
  cat("  stable-baselines3.SAC              -- production SAC\n")
  cat("  cleanrl/sac_continuous_action.py    -- readable reference\n")
  cat("  ray[rllib].SACConfig                -- distributed SAC\n")
  cat("  d3rlpy.SAC / IQL / CQL              -- offline variants\n")
  cat("Extensions:\n")
  cat("  * Learnable temperature alpha (Haarnoja 2018b)\n")
  cat("  * SAC-Discrete (Christodoulou 2019)  -- discrete-action variant\n")
  cat("  * DrQ (Kostrikov 2020), REDQ (Chen 2021), SAC-MoE for improved sample efficiency\n")
}
