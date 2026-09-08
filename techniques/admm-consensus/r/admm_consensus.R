# ADMM & Consensus ADMM (Reference Sec 47.68)
# Native R via ADMM / flsa; Python via cvxpy / pyproximal.
# Run with:  Rscript admm_consensus.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  ADMM                    -- lasso, SCAD, Basis Pursuit, TV via ADMM\n")
  cat("  flsa                    -- fused-lasso via ADMM (Hoefling 2010)\n")
  cat("  glmnet                  -- coord-descent baseline\n")
  cat("Python:\n")
  cat("  cvxpy                   -- automatic ADMM/SCS/OSQP backend selection\n")
  cat("  pyproximal              -- proximal + ADMM operators\n")
  cat("  splitting-methods       -- research code for split algorithms\n")
  cat("  from-scratch            -- see admm_consensus.py\n")
  cat("Refs: Boyd, Parikh, Chu, Peleato & Eckstein (2011) FnT ML 3(1);\n")
  cat("      Parikh & Boyd (2014) 'Proximal algorithms', FnT Optim 1(3).\n")
}
