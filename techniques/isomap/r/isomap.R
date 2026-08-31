# Isomap (Reference Sec 25.14)
# Native R via dimRed / RDRToolbox; Python via sklearn.
# Run with:  Rscript isomap.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  RDRToolbox                   -- Isomap, LLE, Diffusion Maps\n")
  cat("  dimRed                        -- unified manifold-learning wrapper\n")
  cat("  vegan::isomap                 -- ecology-oriented Isomap\n")
  cat("Python:\n")
  cat("  sklearn.manifold.Isomap       -- reference Isomap implementation\n")
  cat("  megaman                        -- large-scale manifold-learning package\n")
  cat("Refs: Tenenbaum, J.B., de Silva, V. & Langford, J.C. (2000) 'A global geometric\n")
  cat("      framework for nonlinear dimensionality reduction', Science 290.\n")
}
