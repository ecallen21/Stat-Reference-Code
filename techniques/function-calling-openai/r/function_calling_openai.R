# Function Calling / Tool Use (Reference Sec 47.204).
#
# OpenAI 2023, Anthropic 2024. Structured JSON tool invocations from
# an LM against a declared tool schema.

if (!requireNamespace("reticulate", quietly = TRUE)) install.packages("reticulate")
library(reticulate)                                   # bridge to Python

cat("=== Function Calling (OpenAI 2023, Anthropic 2024) ===\n")
cat("No native R impl. Recommended libraries (Python):\n")
cat("  * openai (chat.completions.create tools=...)\n")
cat("  * anthropic (messages.create tools=...)\n")
cat("  * langchain / llama-index tool-calling wrappers\n")
