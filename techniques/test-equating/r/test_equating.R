# Test equating (Reference §22.12)
# R via equate::equate.
# Run with:  Rscript test_equating.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  equate::equate(freqtab_Y, freqtab_X, type = 'mean' / 'linear' / 'equipercentile')\n")
  cat("  kequate::keeq2() -- kernel equating (smoothed equipercentile)\n")
  cat("  SNSequate       -- comprehensive equating suite\n")
}
