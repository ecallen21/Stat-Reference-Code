# 2PL / 3PL IRT (Reference §22.6)
# R via ltm (Rizopoulos) or mirt (Chalmers).
# Run with:  Rscript two_three_pl_irt.R

if (sys.nframe() == 0) {
  cat("R packages for 2PL / 3PL:\n")
  cat("  ltm::ltm(Y ~ z1)   -- 2PL via marginal MLE + Gauss-Hermite\n")
  cat("  ltm::tpm(Y)         -- 3PL with pseudo-guessing\n")
  cat("  mirt::mirt(Y, 1, itemtype = '2PL' / '3PL') -- comprehensive IRT toolkit\n")
}
