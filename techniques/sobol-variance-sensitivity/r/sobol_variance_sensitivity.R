# Sobol Sensitivity (Reference Sec 47.334)
# Native R via sensitivity / boot; Python via SALib / UQpy / from-scratch.
# Run with:  Rscript sobol_variance_sensitivity.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  sensitivity::sobol / sobol2002 / sobolSalt   -- Sobol first-order + total\n")
  cat("  sensitivity::sobolmartinez                   -- variance-based\n")
  cat("  fanovaGraph                                  -- FANOVA + Sobol visualisation\n")
  cat("Python:\n")
  cat("  SALib.analyze.sobol\n")
  cat("  UQpy.sensitivity\n")
  cat("  From-scratch (see sobol_variance_sensitivity.py)\n")
  cat("Refs: Sobol, I.M. (1993) 'Sensitivity analysis for non-linear\n")
  cat("      mathematical models', MMCE 1;  Saltelli, A. et al (2008)\n")
  cat("      'Global Sensitivity Analysis: The Primer', Wiley.\n")
}
