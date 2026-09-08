# Plackett-Burman screening design (Reference Sec 47.63)
# Native R via FrF2 / DoE.base; Python via pyDOE / from-scratch.
# Run with:  Rscript plackett_burman_screening.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  FrF2::pb(nruns, nfactors)     -- Plackett-Burman with randomisation\n")
  cat("  DoE.base::oa.design           -- orthogonal-array designs incl. PB\n")
  cat("  unrepx::pnormPlot             -- half-normal contrast plot\n")
  cat("  DoE.wrapper::pbFactorial      -- pipeline wrapper\n")
  cat("Python:\n")
  cat("  pyDOE.pbdesign(k)             -- Plackett-Burman generator\n")
  cat("  pyDOE2                        -- maintained fork with more options\n")
  cat("  from-scratch                  -- see plackett_burman_screening.py\n")
  cat("Refs: Plackett & Burman (1946) Biometrika 33(4); Box, Hunter & Hunter\n")
  cat("      (2005) 'Statistics for Experimenters', 2nd ed, Wiley.\n")
}
