# Group Distributionally Robust Optimisation (Reference Ch 30 Robustness)
# R via reticulate + Python; native R is easy for online-weighted training.
# Run with:  Rscript distributionally_robust_optimization.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  fairml, mlr3fairness         -- group-aware learners incl. weighted variants\n")
  cat("  WeightIt                     -- balancing-weight estimators (adjacent method)\n")
  cat("Python:\n")
  cat("  wilds                        -- Sagawa lab reference DRO for real datasets\n")
  cat("  fairlearn                    -- ExponentiatedGradient / GridSearch fairness algos\n")
  cat("  responsibly, aif360          -- broader fairness / DRO toolkits\n")
  cat("Refs: Sagawa, S., Koh, P.W., Hashimoto, T. & Liang, P. (2020)\n")
  cat("      'Distributionally Robust Neural Networks for Group Shifts', ICLR;\n")
  cat("      Duchi, J. & Namkoong, H. (2019) 'Learning Models with Uniform Performance\n")
  cat("      via Distributionally Robust Optimisation', arXiv:1810.08750.\n")
}
