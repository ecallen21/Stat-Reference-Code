# Gap statistic (Reference Sec 47.81)
# Native R via cluster / factoextra; Python via gap-statistic package.
# Run with:  Rscript gap_statistic_cluster.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  cluster::clusGap             -- reference implementation (Tibshirani)\n")
  cat("  factoextra::fviz_gap_stat    -- ggplot visualisation of Gap and se\n")
  cat("  NbClust                      -- Gap alongside 29 other cluster indices\n")
  cat("Python:\n")
  cat("  gap-statistic                -- PyPI reference package\n")
  cat("  yellowbrick.cluster.KElbow(metric='gap')  -- workflow wrapper\n")
  cat("  from-scratch                 -- see gap_statistic_cluster.py\n")
  cat("Refs: Tibshirani, Walther & Hastie (2001) 'Estimating the number of\n")
  cat("      clusters in a data set via the gap statistic', JRSS-B 63(2).\n")
}
