# Detrended Fluctuation Analysis (Reference Sec 47.330)
# Native R via nonlinearTseries / fractal / pracma; Python via nolds / MFDFA / from-scratch.
# Run with:  Rscript dfa_hurst_fluctuation.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  nonlinearTseries::dfa   -- DFA + multifractal extensions\n")
  cat("  fractal::DFA            -- classical Hurst estimation\n")
  cat("  pracma::hurstexp        -- Hurst-Bartels-Corrado\n")
  cat("Python:\n")
  cat("  nolds.dfa\n")
  cat("  MFDFA (multifractal DFA)\n")
  cat("  antropy.dfa\n")
  cat("  From-scratch (see dfa_hurst_fluctuation.py)\n")
  cat("Refs: Peng, C.-K., Buldyrev, S.V., Havlin, S., Simons, M., Stanley, H.E.\n")
  cat("      & Goldberger, A.L. (1994) 'Mosaic organization of DNA\n")
  cat("      nucleotides', Phys Rev E 49(2).\n")
}
