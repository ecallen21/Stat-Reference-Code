# HDBSCAN clustering (Reference Sec 47.79)
# Native R via dbscan; Python via hdbscan / sklearn.
# Run with:  Rscript hdbscan_clustering.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  dbscan::hdbscan(x, minPts)     -- Campello-Moulavi-Sander HDBSCAN\n")
  cat("  dbscan::dbscan / optics        -- DBSCAN / OPTICS baselines\n")
  cat("  fpc::dbscan                    -- classical DBSCAN alternative\n")
  cat("Python:\n")
  cat("  hdbscan.HDBSCAN                -- McInnes maintained reference\n")
  cat("  sklearn.cluster.HDBSCAN        -- ships in sklearn 1.3+\n")
  cat("  from-scratch reference         -- see hdbscan_clustering.py\n")
  cat("Refs: Campello, Moulavi & Sander (2013) PAKDD; McInnes, Healy & Astels\n")
  cat("      (2017) 'hdbscan' JOSS 2(11).\n")
}
