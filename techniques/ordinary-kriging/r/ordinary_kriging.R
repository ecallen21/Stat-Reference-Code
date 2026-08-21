# Ordinary kriging (Reference §23.7)
# R via gstat.
# Run with:  Rscript ordinary_kriging.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  gstat::krige(z ~ 1, locations = train_sf, newdata = grid_sf, model = variogram_model)\n")
  cat("  geoR::krige.conv                            -- alternative\n")
}
