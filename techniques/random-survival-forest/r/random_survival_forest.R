# Random survival forest (Reference Sec 11.24)
# Native R via randomForestSRC; Python via scikit-survival.
# Run with:  Rscript random_survival_forest.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  randomForestSRC::rfsrc     -- reference RSF (multivariate, competing risks)\n")
  cat("  ranger::ranger(splitrule='logrank')\n")
  cat("  survivalsvm                -- SVM alternative for right-censored data\n")
  cat("Python:\n")
  cat("  sksurv.ensemble.RandomSurvivalForest   -- pip install scikit-survival\n")
  cat("  pysurvival.models.survival_forest\n")
  cat("Refs: Ishwaran, H., Kogalur, U.B., Blackstone, E.H. & Lauer, M.S. (2008)\n")
  cat("      'Random survival forests', Ann Appl Stat 2(3): 841-860.\n")
}
