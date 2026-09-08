# Spatial diff-in-differences (Reference Sec 15.43)
# Native R via spatialreg / spdep / plm; Python via PySAL / linearmodels.
# Run with:  Rscript spatial_diff_in_diff.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  spatialreg + spdep         -- SLX / SAR / SDM models with W weights\n")
  cat("  plm (panel) + spml         -- spatial panel: SDM / SAR panel\n")
  cat("  splm                        -- spatial panel maximum likelihood\n")
  cat("  fixest::feols(., cluster)   -- HAC / cluster SE for spatial FE panels\n")
  cat("Python:\n")
  cat("  PySAL (libpysal / spreg)    -- SLX, SAR, SDM cross-section\n")
  cat("  linearmodels.panel         -- FE panel + custom W\n")
  cat("  From-scratch numpy (see spatial_diff_in_diff.py)\n")
  cat("Refs: Delgado, M.S. & Florax, R.J. (2015) 'Difference-in-differences\n")
  cat("      techniques for spatial data: local autocorrelation and spatial\n")
  cat("      interaction', Econ Lett 137; Chagas, A.L. et al. (2016) 'A spatial\n")
  cat("      difference-in-differences analysis', PLOS ONE 11(3).\n")
}
