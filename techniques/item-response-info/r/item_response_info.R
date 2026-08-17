# Item + test information functions (Reference §22.14)
# R via mirt::testinfo / plot(fit, type = "info").
# Run with:  Rscript item_response_info.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  mirt::testinfo(fit, Theta)  -- test information curve\n")
  cat("  mirt::iteminfo(x, Theta)     -- per-item info\n")
  cat("  plot(mirt_fit, type = 'info')\n")
  cat("  catR                          -- computer adaptive testing based on info\n")
}
