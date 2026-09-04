# Shannon entropy (Reference Sec 34.1)
# Native R via entropy / infotheo; Python via scipy.
# Run with:  Rscript shannon_entropy.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  entropy                       -- Hausser-Strimmer shrinkage entropy\n")
  cat("  infotheo                      -- discrete + continuous MI/entropy\n")
  cat("  FNN                            -- k-NN entropy (Kozachenko-Leonenko)\n")
  cat("Python:\n")
  cat("  scipy.stats.entropy            -- discrete Shannon\n")
  cat("  scipy.stats.differential_entropy -- Vasicek/Van-Es differential entropy\n")
  cat("  NPEET                          -- KSG-family estimators (Kraskov-Stogbauer-Grassberger)\n")
  cat("Refs: Shannon, C.E. (1948) 'A mathematical theory of communication', Bell Sys Tech J;\n")
  cat("      Kozachenko, L.F. & Leonenko, N.N. (1987) 'Sample estimate of the entropy\n")
  cat("      of a random vector', Probl Inf Transm.\n")
}
