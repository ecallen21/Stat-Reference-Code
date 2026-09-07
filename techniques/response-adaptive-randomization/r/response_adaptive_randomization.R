# Response-adaptive randomization (RAR) (Reference Sec 44.13)
# Native R via adaptr / BAR; Python via custom.
# Run with:  Rscript response_adaptive_randomization.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  adaptr                       -- simulation framework for Bayesian adaptive trials\n")
  cat("  BAR (Bayesian Adaptive R)    -- direct Thompson / posterior-based RAR\n")
  cat("  pipe                          -- platform-trial / adaptive-design engine\n")
  cat("  gsDesign                     -- group-sequential (complementary)\n")
  cat("Python:\n")
  cat("  From-scratch numpy (see response_adaptive_randomization.py)\n")
  cat("  adaptr wrapper via rpy2\n")
  cat("  pymc + custom loop for full Bayesian adaptive designs\n")
  cat("Refs: Wei, L.J. & Durham, S. (1978) 'The randomized play-the-winner rule\n")
  cat("      in medical trials', JASA 73(364): 840-843; Berry, D.A. et al.\n")
  cat("      (2010) Bayesian Adaptive Methods for Clinical Trials, CRC.\n")
}
