# Network motifs (Reference §24.11)
# R via igraph or motifr.
# Run with:  Rscript network_motifs.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  igraph::triad_census(g)          -- 16-category directed triad census (Holland-Leinhardt)\n")
  cat("  igraph::motifs(g, size=3 | 4)    -- motif counts up to isomorphism\n")
  cat("  igraph::rewire(g, keeping_degseq(loops=FALSE, niter=k*m))  -- degree-preserving null\n")
  cat("  motifr::compare_motif(...)       -- multi-level networks\n")
  cat("Python: graph_tool::motifs / mfinder (FANMOD) / snap.py for larger motifs.\n")
}
