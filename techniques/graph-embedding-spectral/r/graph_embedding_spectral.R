# Spectral graph embedding (Reference §24.9)
# R via igraph + specc or graphon.
# Run with:  Rscript graph_embedding_spectral.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  igraph::embed_laplacian_matrix(g, no=d, type='I-DAD')  -- normalized Laplacian embedding\n")
  cat("  igraph::embed_adjacency_matrix(g, no=d)                -- adjacency spectral embedding\n")
  cat("  kernlab::specc(x, centers=k)                           -- spectral clustering\n")
  cat("Python:\n")
  cat("  sklearn.manifold.SpectralEmbedding                     -- normalized Laplacian eigenmaps\n")
  cat("  gem / node2vec / stellargraph / pytorch-geometric      -- learned / neural embeddings\n")
}
