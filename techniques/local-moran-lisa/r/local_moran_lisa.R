# Local Moran's I / LISA (Reference §23.4)
# R via spdep::localmoran or rgeoda::local_moran.
# Run with:  Rscript local_moran_lisa.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  spdep::localmoran(x, listw)  -- local Moran I + p (analytic + perm)\n")
  cat("  rgeoda::local_moran(w, x)     -- GeoDa-style LISA cluster map\n")
}
