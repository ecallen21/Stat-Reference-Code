# Variogram modelling (Reference §23.6)
# R via gstat.
# Run with:  Rscript variogram_modeling.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  gstat::variogram(z ~ 1, data = sp_points) -- empirical semivariogram\n")
  cat("  gstat::fit.variogram(vg, vgm(psill, model, range, nugget))\n")
  cat("  geoR::variog / variofit                    -- alternative\n")
}
