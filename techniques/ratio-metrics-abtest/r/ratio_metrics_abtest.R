# Ratio metrics + delta method for A/B tests (Reference Sec 44.10)
# Native R via msm::deltamethod, boot; Python custom + scipy.
# Run with:  Rscript ratio_metrics_abtest.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  msm::deltamethod                 -- delta-method SE for functions of estimates\n")
  cat("  boot                              -- bootstrap SE alternative\n")
  cat("  sandwich                          -- robust SEs / GEE-style clustered ratios\n")
  cat("Python:\n")
  cat("  scipy.stats                       -- t / normal reference distributions\n")
  cat("  statsmodels + custom               -- ratio metrics + delta / bootstrap\n")
  cat("Refs: Deng, Knoblich & Lu (2018) 'Applying the delta method in metric analytics:\n")
  cat("      a practical guide with novel ideas', KDD.\n")
}
