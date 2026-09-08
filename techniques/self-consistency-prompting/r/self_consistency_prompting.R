# Self-consistency prompting (Reference Sec 47.91)
# LLM-only workflow; Python via litellm / lmql / vllm.
# Run with:  Rscript self_consistency_prompting.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  no first-class LLM prompting suite -- roll your own via httr\n")
  cat("  chatgpt / openai              -- OpenAI API bindings\n")
  cat("  gemini.R / groqR              -- alternative LLM providers\n")
  cat("Python:\n")
  cat("  litellm / openai              -- unified LLM API + batch sampling\n")
  cat("  lmql                          -- constrained decoding with sampling\n")
  cat("  vllm                          -- high-throughput batched inference\n")
  cat("  guidance / dspy               -- self-consistency workflows\n")
  cat("  from-scratch                  -- see self_consistency_prompting.py\n")
  cat("Refs: Wang, Wei, Schuurmans, Le, Chi, Narang, Chowdhery & Zhou (2023)\n")
  cat("      'Self-consistency improves chain of thought reasoning', ICLR.\n")
}
