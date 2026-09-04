# Oaxaca-Blinder decomposition (Reference Sec 35.21)
# Native R via oaxaca; Python via pip pkg.
# Run with:  Rscript oaxaca_blinder.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  oaxaca                       -- Hlavac reference (2-fold + 3-fold + subgroup)\n")
  cat("  ineq                          -- adjacent inequality decompositions\n")
  cat("Python:\n")
  cat("  oaxaca-blinder                -- Python port\n")
  cat("  statsmodels + custom OLS      -- manual\n")
  cat("Refs: Blinder, A.S. (1973) 'Wage discrimination: reduced form and structural\n")
  cat("      estimates', J Human Resources; Oaxaca, R. (1973) 'Male-female wage\n")
  cat("      differentials in urban labor markets', Int Econ Rev.\n")
}
