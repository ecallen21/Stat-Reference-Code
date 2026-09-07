# Active learning -- pool-based query strategies (Reference Sec 47.27)
# Native R via ALEval / activeselector; Python via modAL / small-text.
# Run with:  Rscript active_learning_query_strategies.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  ALEval                    -- active-learning evaluation over labels\n")
  cat("  activeselector             -- pool-based query strategies\n")
  cat("  caret + custom loop        -- roll uncertainty / margin queries\n")
  cat("Python:\n")
  cat("  modAL                     -- pool + stream-based active learning\n")
  cat("  small-text                 -- text-classification active learning\n")
  cat("  scikit-learn + custom acquisition\n")
  cat("  ALiPy                      -- comprehensive AL framework\n")
  cat("Refs: Settles, B. (2010) Active Learning Literature Survey, U Wisconsin\n")
  cat("      CS Tech Report 1648; Cohn, D., Atlas, L. & Ladner, R. (1994)\n")
  cat("      'Improving generalization with active learning', Machine Learning.\n")
}
