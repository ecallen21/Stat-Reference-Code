# Interference + cluster randomization (Reference Sec 44.6)
# Native R via inferference / DeclareDesign; Python custom + rpy2.
# Run with:  Rscript interference_cluster.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  inferference                     -- causal inference with interference\n")
  cat("  DeclareDesign                     -- pre-registered design specification\n")
  cat("  clusterSEs                        -- cluster-robust SEs\n")
  cat("  lme4                               -- mixed-effects analysis of clustered data\n")
  cat("Python:\n")
  cat("  linearmodels.iv (cluster-robust SEs)\n")
  cat("  statsmodels (cluster-robust SEs in OLS)\n")
  cat("  custom cluster-mean t-test\n")
  cat("Refs: Blake & Coey (2014) 'Why marketplace experimentation is harder than it\n")
  cat("      seems', EC; Saveski, Pouget-Abadie, Saint-Jacques, Duan, Ghosh, Xu &\n")
  cat("      Airoldi (2017) 'Detecting network effects: randomizing over randomized\n")
  cat("      experiments', KDD.\n")
}
