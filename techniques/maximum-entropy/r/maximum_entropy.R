# Maximum entropy (Reference Sec 34.9)
# Native R via maxentropy / infotheo; Python via scipy.optimize.
# Run with:  Rscript maximum_entropy.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  maxentropy                   -- MaxEnt with moment constraints\n")
  cat("  ENiRG                         -- ecological MaxEnt niche models\n")
  cat("  dismo                          -- species-distribution MaxEnt (Maxent 3.4)\n")
  cat("Python:\n")
  cat("  scipy.optimize + Lagrangian    -- custom MaxEnt solver\n")
  cat("  maxentpy / maxent-classifier   -- feature-based text MaxEnt\n")
  cat("  Elapid                         -- MaxEnt species-distribution modelling\n")
  cat("Refs: Jaynes, E.T. (1957) 'Information theory and statistical mechanics',\n")
  cat("      Physical Review; Cover, T.M. & Thomas, J.A. (2006) 'Elements of\n")
  cat("      Information Theory', Wiley, Ch. 12.\n")
}
