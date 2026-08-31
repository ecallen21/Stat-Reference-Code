# Exploration strategies (Reference §28.11)
# R via reticulate + Python.
# Run with:  Rscript exploration_strategies.R

if (sys.nframe() == 0) {
  cat("R packages: minimal native R; use reticulate + Python.\n")
  cat("Python:\n")
  cat("  stable-baselines3 (with EpsilonSchedule, Noisy Nets)\n")
  cat("  cleanrl -- reference implementations of intrinsic-motivation methods\n")
  cat("  RND (Burda 2018), ICM (Pathak 2017), NGU (Badia 2020), Agent57 (Badia 2020)\n")
  cat("Families:\n")
  cat("  * Undirected: eps-greedy, Boltzmann, decaying-eps\n")
  cat("  * Optimism: UCB, UCB-V, KL-UCB, MBIE-EB\n")
  cat("  * Posterior sampling: Thompson (see multi-armed-bandits), Bootstrapped-DQN\n")
  cat("  * Count-based intrinsic: sqrt(1/N(s)) bonus; pseudo-counts via density models\n")
  cat("  * Prediction-error intrinsic: ICM (forward model), RND (random-network target)\n")
  cat("  * Empowerment / mutual-information: DIAYN, Diversity-is-All-You-Need\n")
  cat("  * Go-Explore (Ecoffet 2019): remember + return to promising states.\n")
}
