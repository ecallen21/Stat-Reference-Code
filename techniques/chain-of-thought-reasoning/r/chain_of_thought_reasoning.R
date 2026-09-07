# Chain-of-thought (CoT) reasoning + self-consistency (Reference Sec 47.26)
# Prompting technique; Python via HuggingFace / OpenAI / Anthropic APIs.
# Run with:  Rscript chain_of_thought_reasoning.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  ellmer                    -- Posit LLM client (Claude, OpenAI, Gemini)\n")
  cat("  chattr                    -- shell / conversational LLM wrappers\n")
  cat("  gptstudio                 -- RStudio addin for prompted completions\n")
  cat("Python:\n")
  cat("  openai / anthropic / google-generativeai -- CoT via API\n")
  cat("  langchain / dspy -- CoT / ReAct / structured prompting frameworks\n")
  cat("  lm-eval-harness (EleutherAI) -- CoT / SC benchmark evaluation\n")
  cat("Refs: Wei, J. et al. (2022) 'Chain-of-thought prompting elicits reasoning\n")
  cat("      in large language models', NeurIPS; Wang, X. et al. (2022) 'Self-\n")
  cat("      consistency improves chain of thought reasoning in language\n")
  cat("      models', ICLR; Kojima, T. et al. (2022) 'Large language models\n")
  cat("      are zero-shot reasoners', NeurIPS.\n")
}
