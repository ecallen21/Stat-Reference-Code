# Graph Convolutional Network (Reference §27.x extra)
# R via torch + custom GCN or via reticulate + PyTorch Geometric / DGL.
# Run with:  Rscript graph_neural_network.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  torch (manual GCN layer: y = A_hat_sym * X * W)\n")
  cat("  reticulate + torch_geometric (torch_geometric.nn.GCNConv / GATConv / SAGEConv)\n")
  cat("  reticulate + dgl (dgl.nn.GraphConv / GATConv / SAGEConv)\n")
  cat("Common architectures:\n")
  cat("  * GCN (Kipf-Welling 2017) — symmetric-normalised mean aggregation\n")
  cat("  * GraphSAGE (Hamilton-Ying-Leskovec 2017) — sampled inductive aggregation\n")
  cat("  * GAT (Velickovic 2018) — attention over neighbours\n")
  cat("  * GIN (Xu 2019) — MLP-based sum aggregation; maximally expressive within WL-1\n")
  cat("  * MPNN framework (Gilmer 2017), Graph Transformer, GraphGPS, GRIT.\n")
}
