# Diffusion maps (Reference Sec 25.16)
# Native R via diffusionMap; Python via pyDiffMap / datafold.
# Run with:  Rscript diffusion_maps.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  diffusionMap                 -- Richards et al. reference implementation\n")
  cat("  RDRToolbox                   -- Isomap + LLE + diffusion maps in one package\n")
  cat("  destiny (Bioconductor)        -- single-cell diffusion pseudotime\n")
  cat("Python:\n")
  cat("  pyDiffMap                     -- Coifman-Lafon reference\n")
  cat("  datafold                      -- diffusion maps + kernel-based DR framework\n")
  cat("  scanpy.tl.diffmap             -- single-cell diffusion component analysis\n")
  cat("Refs: Coifman, R.R. & Lafon, S. (2006) 'Diffusion maps', Applied and\n")
  cat("      Computational Harmonic Analysis;\n")
  cat("      Nadler, B., Lafon, S., Coifman, R.R. & Kevrekidis, I.G. (2006) 'Diffusion\n")
  cat("      maps, spectral clustering and reaction coordinates of dynamical systems',\n")
  cat("      Applied and Computational Harmonic Analysis.\n")
}
