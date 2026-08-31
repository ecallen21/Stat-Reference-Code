# Functional clustering (Reference Sec 31.6)
# Native R via funHDDC / fda.usc; Python via scikit-fda.
# Run with:  Rscript functional_clustering.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  funHDDC                      -- Bouveyron-Jacques model-based functional mixtures\n")
  cat("  fda.usc::kmeans.fd            -- functional k-means with L2 distance\n")
  cat("  Funclustering                 -- Model-based clustering with FPCA\n")
  cat("  fdapace                        -- PACE-based clustering for sparse curves\n")
  cat("Python:\n")
  cat("  scikit-fda                    -- k-means on FDataGrid with L2 metric\n")
  cat("  tslearn                       -- DTW-based k-means for time-series curves\n")
  cat("Refs: James, G.M. & Sugar, C.A. (2003) 'Clustering for sparsely sampled\n")
  cat("      functional data', JASA; Bouveyron, C. & Jacques, J. (2011) 'Model-based\n")
  cat("      clustering of time series in group-specific functional subspaces', ADAC.\n")
}
