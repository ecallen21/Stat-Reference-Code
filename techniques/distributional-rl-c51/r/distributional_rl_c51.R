# C51 distributional RL (Reference Sec 47.124)
# Python via dopamine / stable-baselines3-contrib / torchrl.
# Run with:  Rscript distributional_rl_c51.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  no first-class RL library in R; reticulate wraps Python engines\n")
  cat("Python:\n")
  cat("  dopamine.agents.rainbow      -- Google Dopamine C51 / Rainbow\n")
  cat("  stable-baselines3-contrib.QRDQN -- QR-DQN alternative\n")
  cat("  torchrl.modules.QValueModule / DistributionalQ  -- TorchRL primitives\n")
  cat("  cleanrl                       -- single-file distributional RL demos\n")
  cat("  from-scratch                  -- see distributional_rl_c51.py\n")
  cat("Refs: Bellemare, Dabney & Munos (2017) 'A distributional perspective on\n")
  cat("      RL', ICML; Dabney et al (2018) 'Implicit quantile networks', ICML.\n")
}
