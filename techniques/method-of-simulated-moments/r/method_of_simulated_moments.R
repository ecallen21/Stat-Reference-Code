# Method of Simulated Moments (MSM) (Reference Sec 45.7)
# Native R via gmm / momentfit; Python via PyBLP + custom.
# Run with:  Rscript method_of_simulated_moments.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  gmm::gmm(..., type = 'iterative')  -- simulated / iterative GMM\n")
  cat("  momentfit::momentModel + gmmFit    -- fresh S4 GMM framework\n")
  cat("  BLPestimatoR                        -- discrete-choice / demand estimation\n")
  cat("  msm                                 -- not this MSM (multi-state models)\n")
  cat("Python:\n")
  cat("  pyblp                              -- Berry-Levinsohn-Pakes demand\n")
  cat("  statsmodels.sandbox.regression.gmm -- MSM as generalised GMM\n")
  cat("  linearmodels.iv.IVGMM              -- moments-based estimation\n")
  cat("Refs: McFadden, D. (1989) 'A method of simulated moments for estimation\n")
  cat("      of discrete response models without numerical integration',\n")
  cat("      Econometrica 57(5): 995-1026; Pakes, A. & Pollard, D. (1989)\n")
  cat("      'Simulation and the asymptotics of optimization estimators',\n")
  cat("      Econometrica 57(5): 1027-1057.\n")
}
