# Deep Sets (Reference Sec 47.312)
# Native R via torch (R) + custom; Python via PyG.aggr / torchset / from-scratch.
# Run with:  Rscript deepset_permutation_invariant.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  torch (R) with rowwise MLP + sum / mean pool\n")
  cat("  reticulate + torch-geometric\n")
  cat("Python:\n")
  cat("  torch_geometric.nn.aggr (Sum/Mean/Max Aggregation)\n")
  cat("  set-transformer (Lee et al 2019) - self-attention alternative\n")
  cat("  From-scratch (see deepset_permutation_invariant.py)\n")
  cat("Refs: Zaheer, M., Kottur, S., Ravanbakhsh, S., Poczos, B.,\n")
  cat("      Salakhutdinov, R.R. & Smola, A.J. (2017) 'Deep sets', NeurIPS.\n")
}
