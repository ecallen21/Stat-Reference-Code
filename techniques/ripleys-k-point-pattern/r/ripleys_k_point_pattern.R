# Ripley's K / L functions for spatial point patterns (Reference §23.12)
# R via spatstat.
# Run with:  Rscript ripleys_k_point_pattern.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  spatstat.explore::Kest(ppp, correction='iso')   -- Ripley's isotropic edge correction\n")
  cat("  spatstat.explore::Lest(ppp)                     -- L-transform (variance-stabilised)\n")
  cat("  spatstat.explore::envelope(ppp, Kest, nsim=99)  -- Monte-Carlo CSR envelope\n")
  cat("  spatstat.explore::pcf(ppp)                      -- pair correlation function\n")
}
