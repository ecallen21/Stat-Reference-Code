# Silhouette clustering validation (Reference Sec 47.74)
# Native R via cluster; Python via sklearn.
# Run with:  Rscript silhouette_clusters.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  cluster::silhouette / summary(silhouette(...))  -- Rousseeuw reference\n")
  cat("  factoextra::fviz_silhouette                     -- ggplot silhouette plot\n")
  cat("  NbClust::NbClust                                -- 30 validity indices, incl silhouette\n")
  cat("Python:\n")
  cat("  sklearn.metrics.silhouette_score / silhouette_samples\n")
  cat("  yellowbrick.cluster.SilhouetteVisualizer         -- plot per-cluster distributions\n")
  cat("  from-scratch                                     -- see silhouette_clusters.py\n")
  cat("Refs: Rousseeuw (1987) J Comp Appl Math 20; Kaufman & Rousseeuw (1990)\n")
  cat("      'Finding Groups in Data', Wiley.\n")
}
