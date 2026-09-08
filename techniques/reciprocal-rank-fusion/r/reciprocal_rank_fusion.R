# Reciprocal Rank Fusion (Reference Sec 47.131)
# Trivial to implement in R; Python via ranx / pytrec_eval.
# Run with:  Rscript reciprocal_rank_fusion.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  no established RRF package -- roll your own in a few lines\n")
  cat("Python:\n")
  cat("  ranx.fuse(rankings, method='rrf')  -- retrieval-eval + fusion library\n")
  cat("  pyserini.fusion                     -- IR fusion (RRF, CombSUM, CombMNZ)\n")
  cat("  from-scratch                        -- see reciprocal_rank_fusion.py\n")
  cat("Refs: Cormack, Clarke & Buttcher (2009) 'Reciprocal rank fusion outperforms\n")
  cat("      Condorcet and individual rank learning methods', SIGIR.\n")
}
