# LinUCB Contextual Bandit (Reference Sec 47.305)
# Native R via contextual / MABWiser via reticulate; Python via contextualbandits / from-scratch.
# Run with:  Rscript linucb_contextual_bandit.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  contextual                 -- contextual bandit simulator with LinUCB\n")
  cat("  reticulate + MABWiser / contextualbandits\n")
  cat("  bandit / DiceKriging       -- GP-bandit variants\n")
  cat("Python:\n")
  cat("  contextualbandits          -- LinUCB, LinTS, LogisticUCB\n")
  cat("  MABWiser                   -- broad bandit library\n")
  cat("  vowpalwabbit --cb           -- contextual bandit at scale\n")
  cat("  From-scratch (see linucb_contextual_bandit.py)\n")
  cat("Refs: Li, L., Chu, W., Langford, J. & Schapire, R.E. (2010) 'A\n")
  cat("      contextual-bandit approach to personalized news article\n")
  cat("      recommendation', WWW.\n")
}
