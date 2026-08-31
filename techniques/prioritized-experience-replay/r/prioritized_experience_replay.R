# Prioritised experience replay (Reference §28.x extra)
# R via reticulate + Python.
# Run with:  Rscript prioritized_experience_replay.R

if (sys.nframe() == 0) {
  cat("R packages: use reticulate + Python.\n")
  cat("Python:\n")
  cat("  stable-baselines3-contrib.PrioritizedReplayBuffer (in offline / off-policy trainers)\n")
  cat("  ray[rllib] PrioritizedReplayBuffer configured via config\n")
  cat("  cleanrl/dqn_atari_per.py -- readable single-file PER + DQN\n")
  cat("Data structures:\n")
  cat("  * Sum-tree (O(log N) sampling)          -- the standard efficient implementation\n")
  cat("  * Segment-tree / Fenwick tree           -- alternatives\n")
  cat("Extensions:\n")
  cat("  * Rank-based PER (Schaul 2016)          -- rank of |TD| instead of magnitude\n")
  cat("  * Hindsight Experience Replay (HER)     -- relabels goals; complementary to PER\n")
  cat("  * Prioritised sequence replay (R2D2)    -- prioritise sequences, not transitions\n")
}
