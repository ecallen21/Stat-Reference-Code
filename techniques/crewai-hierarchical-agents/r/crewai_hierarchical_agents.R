# CrewAI (Reference Sec 47.208).
#
# Moura 2024 (CrewAI OSS). Multi-agent workflows with typed Task
# outputs and sequential / hierarchical processes.

if (!requireNamespace("reticulate", quietly = TRUE)) install.packages("reticulate")
library(reticulate)                                   # bridge to Python

cat("=== CrewAI (Moura 2024) ===\n")
cat("No native R impl. Recommended libraries (Python):\n")
cat("  * crewai (main package)\n")
cat("  * crewai-tools (built-in tool library)\n")
cat("  * langgraph / autogen as alternatives\n")
