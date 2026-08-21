# Spatial weights matrix (Reference §23.2)
# R via spdep (Bivand).
# Run with:  Rscript spatial_weights_matrix.R

if (sys.nframe() == 0) {
  cat("R packages for spatial weights:\n")
  cat("  spdep::poly2nb(polygons)               -- rook / queen contiguity\n")
  cat("  spdep::knn2nb(knearneigh(coords, k))   -- kNN\n")
  cat("  spdep::dnearneigh(coords, 0, d)        -- distance band\n")
  cat("  spdep::nb2listw(nb, style = 'W')       -- row-standardized listw\n")
}
