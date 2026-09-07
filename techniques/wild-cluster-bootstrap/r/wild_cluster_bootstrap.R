# Wild cluster bootstrap (Reference Sec 12.15)
# Native R via fwildclusterboot / clubSandwich; Python via wildboottest.
# Run with:  Rscript wild_cluster_bootstrap.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  fwildclusterboot::boottest  -- fast wild-cluster bootstrap (Roodman et al.)\n")
  cat("  clubSandwich                -- CR1 / CR2 cluster-robust SEs\n")
  cat("  lmtest / sandwich           -- CR alternatives (vcovCR)\n")
  cat("  multiwayvcov::cluster.boot   -- pairs cluster bootstrap\n")
  cat("Python:\n")
  cat("  wildboottest (pip)          -- Python port of fwildclusterboot\n")
  cat("  linearmodels.iv.IV2SLS with cov_type='clustered'  -- CR sandwich\n")
  cat("Refs: Cameron, A.C., Gelbach, J.B. & Miller, D.L. (2008) 'Bootstrap-based\n")
  cat("      improvements for inference with clustered errors', REStat 90(3);\n")
  cat("      MacKinnon, J.G. & Webb, M.D. (2018) 'The wild bootstrap for few\n")
  cat("      (treated) clusters', Econom J 21(2): 114-135.\n")
}
