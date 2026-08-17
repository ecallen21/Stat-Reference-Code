# Samejima's Graded Response Model (Reference §22.7)
# R via mirt::mirt(Y, 1, itemtype = "graded") or ltm::grm.
# Run with:  Rscript graded_response_model.R

if (sys.nframe() == 0) {
  cat("R packages for GRM:\n")
  cat("  ltm::grm(Y)                             -- classical GRM MML\n")
  cat("  mirt::mirt(Y, 1, itemtype = 'graded')   -- modern comprehensive IRT\n")
}
