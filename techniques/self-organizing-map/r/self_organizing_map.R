# Self-Organizing Map (Reference Sec 47.311)
# Native R via kohonen / som; Python via minisom / from-scratch.
# Run with:  Rscript self_organizing_map.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  kohonen                    -- SOM with U-matrix and supervised variants\n")
  cat("  som                        -- classical Kohonen SOM in R\n")
  cat("  ClustGeo / trimcluster    -- alternative topology-preserving clusters\n")
  cat("Python:\n")
  cat("  minisom                    -- lightweight numpy SOM\n")
  cat("  susi (Riese)               -- supervised + unsupervised SOM\n")
  cat("  From-scratch numpy (see self_organizing_map.py)\n")
  cat("Refs: Kohonen, T. (1982) 'Self-organized formation of topologically\n")
  cat("      correct feature maps', Biological Cybernetics 43;\n")
  cat("      Kohonen, T. (2001) 'Self-Organizing Maps', Springer, 3rd ed.\n")
}
