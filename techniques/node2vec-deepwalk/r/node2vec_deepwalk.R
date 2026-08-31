# Node embeddings: DeepWalk / node2vec (Reference Sec 30.19)
# R via reticulate + Python; native R via node2vec pkg (limited).
# Run with:  Rscript node2vec_deepwalk.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  node2vec                     -- R wrapper around the reference algorithm\n")
  cat("  igraph::embed_adjacency_matrix -- classical spectral embeddings (adjacent)\n")
  cat("Python:\n")
  cat("  node2vec                     -- Grover-Leskovec reference (biased walks + Word2Vec)\n")
  cat("  gensim                        -- Word2Vec on generated walks\n")
  cat("  pytorch-geometric (torch_geometric.nn.Node2Vec) -- GPU-accelerated\n")
  cat("  karateclub                    -- unified node/graph embedding toolkit\n")
  cat("Refs: Perozzi, B., Al-Rfou, R. & Skiena, S. (2014) 'DeepWalk: online learning\n")
  cat("      of social representations', KDD;\n")
  cat("      Grover, A. & Leskovec, J. (2016) 'node2vec: scalable feature learning\n")
  cat("      for networks', KDD.\n")
}
