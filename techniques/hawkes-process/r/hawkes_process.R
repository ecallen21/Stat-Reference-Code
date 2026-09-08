# Hawkes process (Reference Sec 47.41)
# Native R via hawkes / PtProcess; Python via tick / from-scratch.
# Run with:  Rscript hawkes_process.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  hawkes::simulateHawkes / hawkesLikelihood -- univariate & multivariate\n")
  cat("  PtProcess::mpp                             -- more general marked point proc\n")
  cat("  evently                                    -- retweet / cascade Hawkes\n")
  cat("Python:\n")
  cat("  tick.hawkes                                -- exp/pow-law kernels, ADMM fit\n")
  cat("  from-scratch                               -- see hawkes_process.py\n")
  cat("Refs: Hawkes (1971) Biometrika 58; Ogata (1981) IEEE TIT 27; Ozaki (1979)\n")
  cat("      Ann Inst Stat Math 31.\n")
}
