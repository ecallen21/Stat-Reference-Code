# Model-based RL: Dyna-Q + deep MBRL notes (Reference §28.7)
# R via ReinforcementLearning + custom model.
# Run with:  Rscript model_based_rl.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  ReinforcementLearning::ReinforcementLearning + manual model in R\n")
  cat("Python (deep MBRL):\n")
  cat("  World Models (Ha-Schmidhuber 2018)  -- VAE + RNN world model + linear controller\n")
  cat("  MBPO (Janner 2019)                   -- model-based policy optimization with ensembles\n")
  cat("  PETS (Chua 2018)                     -- probabilistic ensemble + trajectory sampling\n")
  cat("  Dreamer v1/v2/v3 (Hafner 2020-2023) -- latent world models with imagination-based training\n")
  cat("  MuZero (Schrittwieser 2020)          -- MCTS in a learned latent model\n")
  cat("  TD-MPC / TD-MPC2 (Hansen 2022)       -- latent model + short-horizon MPC + Q-learning\n")
  cat("Applications: robotics (sample-scarce), scientific control (protein / plasma / chemistry).\n")
}
