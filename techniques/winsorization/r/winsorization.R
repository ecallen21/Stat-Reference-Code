# Winsorization and truncation (Reference Sec 41.5)
# Native R via DescTools; Python scipy.mstats + custom.
# Run with:  Rscript winsorization.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  DescTools::Winsorize             -- symmetric or asymmetric Winsorisation\n")
  cat("  robustHD                         -- robust HD tools inc. Winsor\n")
  cat("  psych::winsor                    -- Winsorised mean/sd\n")
  cat("Python:\n")
  cat("  scipy.stats.mstats.winsorize     -- symmetric / asymmetric Winsorization\n")
  cat("  pandas.Series.clip               -- percentile clipping\n")
  cat("Refs: Wilcox, R.R. (2022) Introduction to Robust Estimation and Hypothesis\n")
  cat("      Testing, 5th ed., Academic Press.\n")
}
