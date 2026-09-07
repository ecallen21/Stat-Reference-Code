# Platform / master-protocol trial design (Reference Sec 44.14)
# Native R via adaptr / pipe / MAMS; Python via custom.
# Run with:  Rscript platform_trial_design.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  adaptr                       -- Bayesian adaptive-platform simulator\n")
  cat("  pipe                          -- comprehensive multi-arm platform engine\n")
  cat("  MAMS                          -- multi-arm multi-stage frequentist designs\n")
  cat("  BOP2                          -- Bayesian optimal Phase II designs\n")
  cat("  octopus                       -- oncology basket-trial engine\n")
  cat("  gsDesign                      -- group-sequential complementary tool\n")
  cat("Python:\n")
  cat("  From-scratch numpy (see platform_trial_design.py)\n")
  cat("  BLADEs / PyBAT (community) -- Bayesian adaptive design frameworks\n")
  cat("  pymc + custom loop -- fully Bayesian designs\n")
  cat("Refs: Woodcock, J. & LaVange, L.M. (2017) 'Master protocols to study\n")
  cat("      multiple therapies, multiple diseases, or both', NEJM 377(1);\n")
  cat("      Berry, S.M. et al. (2015) 'The platform trial: an efficient\n")
  cat("      strategy for evaluating multiple treatments', JAMA 313(16).\n")
}
