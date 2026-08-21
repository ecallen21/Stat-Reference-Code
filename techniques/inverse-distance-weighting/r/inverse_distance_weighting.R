# Inverse Distance Weighting (Reference §23.8)
# R via gstat::krige(z ~ 1, data, newdata, model = NULL, idp = 2).
# Run with:  Rscript inverse_distance_weighting.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  gstat::krige(z ~ 1, data, newdata, model = NULL, idp = 2)  -- IDW\n")
  cat("  gstat::idw(z ~ 1, data, newdata, idp = 2, nmax = 8)         -- convenience wrapper\n")
}
