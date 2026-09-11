# GraphSAGE (Reference Sec 47.149).
#
# Hamilton, Ying & Leskovec 2017. Inductive graph representation
# learning with neighbourhood-sampled AGGREGATOR functions.
#
# No native R implementation; use Python via reticulate.

if (!requireNamespace("reticulate", quietly = TRUE)) install.packages("reticulate")
library(reticulate)                                   # bridge to Python

cat("=== GraphSAGE (Hamilton-Ying-Leskovec 2017) ===\n")
cat("No native R impl. Recommended libraries (Python):\n")
cat("  * pytorch-geometric (torch_geometric.nn.SAGEConv)\n")
cat("  * dgl (dgl.nn.SAGEConv)\n")
