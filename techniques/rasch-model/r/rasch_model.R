# Rasch model (Reference §22.5)
# R via eRm::RM (CML) or ltm::rasch (MML).
# Run with:  Rscript rasch_model.R

if (sys.nframe() == 0) {
  cat("R packages for Rasch modelling:\n")
  cat("  eRm::RM(Y)   -- conditional MLE (CML), no theta distribution assumption\n")
  cat("  ltm::rasch(Y) -- marginal MLE (MML), assumes Normal theta prior\n")
  cat("  TAM::tam.mml  -- more flexible; supports polytomous too\n")
}
