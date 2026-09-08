# Tree of Thoughts (Reference Sec 47.92)
# LLM-only workflow; Python via langchain / tree-of-thoughts.
# Run with:  Rscript tree_of_thoughts.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  no first-class ToT in R -- roll your own around openai bindings\n")
  cat("Python:\n")
  cat("  tree-of-thoughts   -- reference package accompanying Yao et al 2023\n")
  cat("  langchain          -- ToT-style chains + evaluators\n")
  cat("  guidance / dspy    -- structured LLM search patterns\n")
  cat("  from-scratch       -- see tree_of_thoughts.py\n")
  cat("Refs: Yao, Yu, Zhao, Shafran, Griffiths, Cao & Narasimhan (2023)\n")
  cat("      'Tree of Thoughts: Deliberate problem solving with LLMs', NeurIPS.\n")
}
