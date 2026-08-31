# Small-world + scale-free network models (Reference Sec 30.12)
# Native R via igraph; Python via networkx.
# Run with:  Rscript small_world_scale_free.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  igraph::sample_smallworld    -- Watts-Strogatz small-world generator\n")
  cat("  igraph::sample_pa             -- Barabasi-Albert preferential attachment\n")
  cat("  poweRlaw                      -- Clauset-Shalizi-Newman power-law fitting\n")
  cat("Python:\n")
  cat("  networkx.watts_strogatz_graph\n")
  cat("  networkx.barabasi_albert_graph\n")
  cat("  powerlaw                       -- Clauset-Shalizi-Newman fitter\n")
  cat("Refs: Watts, D.J. & Strogatz, S.H. (1998) 'Collective dynamics of small-world\n")
  cat("      networks', Nature;\n")
  cat("      Barabasi, A.-L. & Albert, R. (1999) 'Emergence of scaling in random\n")
  cat("      networks', Science.\n")
}
