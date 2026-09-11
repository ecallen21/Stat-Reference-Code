# Cluster-Randomized Trial (Reference Sec 47.266)
# Native R via clusterPower / lmerTest / geepack; Python via statsmodels.
# Run with:  Rscript cluster_randomized_trial.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  clusterPower               -- SS and power for CRTs across outcomes\n")
  cat("  CRTSize / cluster.rct      -- Donner-Klar SS formulas\n")
  cat("  lmerTest::lmer             -- fit mixed-effects analysis\n")
  cat("  geepack::geeglm            -- GEE with exchangeable / independence\n")
  cat("  ICCbin / ICCbin::iccbin    -- ICC estimation for binary outcomes\n")
  cat("Python:\n")
  cat("  statsmodels.mixed_linear_model\n")
  cat("  linearmodels.PanelOLS (cluster FE)\n")
  cat("  From-scratch cluster-summary + t-test (see cluster_randomized_trial.py)\n")
  cat("Refs: Donner, A. & Klar, N. (2000) 'Design and Analysis of Cluster\n")
  cat("      Randomization Trials in Health Research', Arnold Publishers, London.\n")
}
