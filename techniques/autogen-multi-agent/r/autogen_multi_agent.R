# AutoGen (Reference Sec 47.207).
#
# Wu et al 2023 (Microsoft). Multi-agent conversation framework with
# UserProxy / Assistant / GroupChatManager primitives.

if (!requireNamespace("reticulate", quietly = TRUE)) install.packages("reticulate")
library(reticulate)                                   # bridge to Python

cat("=== AutoGen (Wu et al 2023 Microsoft) ===\n")
cat("No native R impl. Recommended libraries (Python):\n")
cat("  * pyautogen (Microsoft, main package)\n")
cat("  * autogenhub / ag2 (community forks)\n")
cat("  * langgraph for graph-based multi-agent alternative\n")
