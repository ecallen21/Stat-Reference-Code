# Hierarchical RL: options framework (Reference §28.x extra)
# R via reticulate + Python.
# Run with:  Rscript hierarchical_rl_options.R

if (sys.nframe() == 0) {
  cat("R packages: no strong native R support; use reticulate + Python.\n")
  cat("Python:\n")
  cat("  ray[rllib] hierarchical training with option definitions\n")
  cat("  garage / rlkit / cleanrl -- research-quality HRL implementations\n")
  cat("HRL families:\n")
  cat("  * Options framework (Sutton-Precup-Singh 1999) -- temporal abstraction\n")
  cat("  * Feudal networks (Vezhnevets 2017) -- manager + worker with learned goals\n")
  cat("  * Option-critic (Bacon 2017) -- learn options end-to-end via policy gradient\n")
  cat("  * HIRO (Nachum 2018) -- goal-conditioned hierarchy on continuous control\n")
  cat("  * DIAYN (Eysenbach 2019) -- unsupervised skill discovery\n")
  cat("  * MAXQ (Dietterich 2000) -- value-function decomposition over a task hierarchy\n")
}
