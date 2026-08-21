# HITS authority + hub scores (Reference §24.x extra)
# R via igraph.
# Run with:  Rscript hits_authority_hub.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  igraph::authority_score(g)   -- Kleinberg authorities\n")
  cat("  igraph::hub_score(g)         -- Kleinberg hubs\n")
  cat("  igraph::page_rank(g)         -- PageRank (compare/contrast)\n")
  cat("  For Salsa (random-walk HITS), Kleinberg-with-teleportation, use custom power iteration.\n")
  cat("Python: networkx.hits, scipy.sparse.linalg.svds on A for spectral form.\n")
}
