# Offline reinforcement learning (Reference §28.9)
# R via reticulate + Python d3rlpy or ray[rllib] offline.
# Run with:  Rscript offline_rl.R

if (sys.nframe() == 0) {
  cat("R packages: no strong native R package for offline RL; use reticulate + Python.\n")
  cat("Python:\n")
  cat("  d3rlpy       -- unified library: CQL, BCQ, BEAR, IQL, TD3+BC, AWAC, PLAS, EDAC, ...\n")
  cat("  ray[rllib] offline API (Marwil, CQL, BCQ)\n")
  cat("  stable-baselines3-contrib -- some offline algorithms\n")
  cat("Algorithms:\n")
  cat("  * BC (behavioural cloning) -- upper bound of the behaviour policy\n")
  cat("  * BCQ (Fujimoto 2019)      -- conditional VAE + Q ensemble; only choose in-distribution actions\n")
  cat("  * CQL (Kumar 2020)          -- add a pessimism penalty on OOD Q-values\n")
  cat("  * IQL (Kostrikov 2021)      -- expectile regression + advantage-weighted BC (SOTA on D4RL)\n")
  cat("  * TD3+BC (Fujimoto 2021)   -- TD3 + a simple BC regulariser; SOTA-competitive\n")
  cat("  * AWAC (Nair 2020)          -- advantage-weighted actor critic\n")
  cat("Applications: healthcare (dynamic treatment regimes), robotics (log-only fine-tune),\n")
  cat("              finance (backtesting-safe learning from historical trades).\n")
}
