# Compositional data analysis (Reference Sec 38.2)
# Native R via compositions / robCompositions; Python custom + scikit-bio.
# Run with:  Rscript compositional_data.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  compositions::acomp/alr/clr/ilr -- Aitchison geometry primitives\n")
  cat("  robCompositions                 -- robust log-ratio methods\n")
  cat("  zCompositions                   -- zero replacement (multiplicative, log-ratio)\n")
  cat("Python:\n")
  cat("  scikit-bio (skbio.stats.composition: clr, ilr, closure)\n")
  cat("  custom                          -- ALR/CLR/ILR + Aitchison distance\n")
  cat("Refs: Aitchison, J. (1986) The Statistical Analysis of Compositional Data,\n")
  cat("      Chapman & Hall; Pawlowsky-Glahn, Egozcue & Tolosana-Delgado (2015)\n")
  cat("      Modeling and Analysis of Compositional Data, Wiley;\n")
  cat("      Egozcue et al. (2003) 'Isometric logratio transformations', Math Geol.\n")
}
