# Moran's I + Geary's C (Reference §23.3)
# R via spdep::moran.test / geary.test.
# Run with:  Rscript morans_i_gearys_c.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  spdep::moran.test(x, listw)   -- analytic Moran's I test\n")
  cat("  spdep::moran.mc(x, listw, nsim = 999) -- permutation Moran's I\n")
  cat("  spdep::geary.test(x, listw)   -- Geary's C\n")
}
