# Rosenbaum bounds (Reference Sec 15.23)
# Native R via rbounds / sensitivitymv / senmv; Python custom.
# Run with:  Rscript rosenbaum_bounds.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  rbounds::binarysens / hlsens     -- binary + continuous matched-pair bounds\n")
  cat("  sensitivitymv                     -- multivariate Rosenbaum bounds\n")
  cat("  sensitivitymw                     -- matched-set sensitivity\n")
  cat("  EValue                             -- E-value alternative (VanderWeele)\n")
  cat("Python:\n")
  cat("  custom (scipy.stats.binom + Rosenbaum formula)\n")
  cat("  zepid + evalue (sensitivity-e-value already implemented)\n")
  cat("Refs: Rosenbaum, P.R. (2002) Observational Studies, 2nd ed., Springer;\n")
  cat("      Rosenbaum, P.R. (2010) Design of Observational Studies, Springer.\n")
}
