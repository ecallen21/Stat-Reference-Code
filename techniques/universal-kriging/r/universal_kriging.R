# Universal / drift kriging (Reference §23.x extra)
# R via gstat.
# Run with:  Rscript universal_kriging.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  gstat::krige(z ~ x + y, obs_sp, grid_sp, model=vgm(...))     -- universal kriging\n")
  cat("  gstat::krige(z ~ elevation, obs_sp, grid_sp, model=vgm(...)) -- kriging w/ external drift (KED)\n")
  cat("  gstat::variogram(z ~ x + y, obs_sp)                          -- residual variogram after removing drift\n")
  cat("  automap::autoKrige(z ~ x + y, ...)                           -- automatic model+drift selection\n")
  cat("  fields::mKrig / spatialProcess                               -- reproducing-kernel form\n")
}
