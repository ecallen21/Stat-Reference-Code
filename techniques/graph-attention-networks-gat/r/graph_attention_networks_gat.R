# Graph Attention Networks (Reference Sec 47.150).
#
# Velickovic et al 2018. Learned attention weights on graph edges.
#
# No native R implementation; use Python via reticulate.

if (!requireNamespace("reticulate", quietly = TRUE)) install.packages("reticulate")
library(reticulate)                                   # bridge to Python

cat("=== Graph Attention Network - GAT (Velickovic et al 2018) ===\n")
cat("No native R impl. Recommended libraries (Python):\n")
cat("  * pytorch-geometric (torch_geometric.nn.GATConv, GATv2Conv)\n")
cat("  * dgl (dgl.nn.GATConv)\n")
