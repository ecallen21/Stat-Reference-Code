# Disproportionality analysis (Reference Sec 43.1)
# Native R via PhViD; Python vigipy + custom.
# Run with:  Rscript disproportionality_signal_detection.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  PhViD::bcpnn / PRR / ROR / MGPS -- pharmacovigilance disproportionality\n")
  cat("  pvLRT                            -- likelihood-ratio-based signal tests\n")
  cat("Python:\n")
  cat("  vigipy                            -- Uppsala-style pharmacovigilance toolkit\n")
  cat("  custom (scipy.stats chi^2 + BCPNN)\n")
  cat("Refs: Bate, A. & Evans, S.J.W. (2009) 'Quantitative signal detection using\n")
  cat("      spontaneous ADR reporting', Pharmacoepi Drug Saf; DuMouchel, W. (1999)\n")
  cat("      'Bayesian data mining in large frequency tables (MGPS)', Am Stat.\n")
}
