# In-context learning (ICL) (Reference Sec 47.20)
# Native R via ellmer; Python via openai / anthropic / transformers.
# Run with:  Rscript in_context_learning.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  ellmer                    -- Posit LLM client (Claude, OpenAI, Gemini)\n")
  cat("  chattr                    -- shell / conversational LLM wrappers\n")
  cat("  gptstudio                 -- RStudio addin, prompt-driven completions\n")
  cat("Python:\n")
  cat("  openai / anthropic / google-generativeai   -- ICL via API\n")
  cat("  transformers                                 -- Local ICL evaluation\n")
  cat("  lm-eval-harness (EleutherAI)                 -- standardised ICL benchmarks\n")
  cat("Refs: Brown, T. et al. (2020) 'Language models are few-shot learners',\n")
  cat("      NeurIPS 33: 1877-1901; Xie, S.M. et al. (2022) 'An explanation of\n")
  cat("      in-context learning as implicit Bayesian inference', ICLR;\n")
  cat("      Akyurek, E. et al. (2023) 'What learning algorithm is in-context\n")
  cat("      learning?', ICLR.\n")
}
