# Getis-Ord Gi / Gi* hot-spot statistics (Reference §23.x extra)
# R via spdep or rgeoda.
# Run with:  Rscript getis_ord_g_statistic.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  spdep::localG(x, listw, zero.policy=TRUE)   -- Gi* z-scores\n")
  cat("  spdep::localG_perm(x, listw, nsim=999)      -- permutation p-values\n")
  cat("  rgeoda::local_g / local_gstar               -- GeoDa-compatible Getis-Ord\n")
  cat("  tmap::qtm(sf_obj, fill='Gi_star')           -- quick hot-spot map\n")
  cat("Python: pysal / esda::G_Local(x, w, star=True)\n")
}
