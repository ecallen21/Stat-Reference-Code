# SimPO (Reference Sec 47.199).
#
# Meng-Xia-Chen 2024. Reference-free, length-normalised preference
# optimisation with a target margin gamma.

if (!requireNamespace("reticulate", quietly = TRUE)) install.packages("reticulate")
library(reticulate)                                   # bridge to Python

cat("=== SimPO (Meng et al 2024) ===\n")
cat("No native R impl. Recommended libraries (Python):\n")
cat("  * trl.CPOTrainer with SimPO loss variant\n")
cat("  * princeton-nlp/SimPO (reference impl)\n")
cat("  * axolotl SimPO configuration\n")
