# Stochastic Block Model (Reference §24.6)
# R via blockmodels, sbm, or greed.
# Run with:  Rscript stochastic_block_model.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  blockmodels::BM_bernoulli$new('SBM_sym', adj)      -- variational EM\n")
  cat("  sbm::estimateSimpleSBM(adj, 'bernoulli')           -- more general (weighted, bipartite)\n")
  cat("  greed::greed(adj, model=Sbm())                     -- ICL-based model choice for K\n")
  cat("  igraph::sample_sbm(n, pref.matrix, block.sizes)    -- simulator\n")
  cat("Python:  graph_tool.inference.minimize_blockmodel_dl -- nested SBM with MDL model selection\n")
}
