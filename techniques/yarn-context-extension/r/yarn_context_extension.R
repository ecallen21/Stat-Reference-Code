# YaRN context extension (Reference Sec 47.49)
# LLM-specific; Python has all the tooling.
# Run with:  Rscript yarn_context_extension.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  no established R implementation -- fine-tuning LLMs happens in Python\n")
  cat("Python:\n")
  cat("  transformers   -- HF LlamaConfig(rope_scaling={'type':'yarn', ...})\n")
  cat("  vllm           -- inference engine with YaRN scaling knobs\n")
  cat("  llama.cpp      -- llama-cli --rope-scaling yarn / freq-scale / attn-scale\n")
  cat("  yarn (GitHub)  -- reference implementation by Peng, Quesnelle et al.\n")
  cat("  from-scratch   -- see yarn_context_extension.py\n")
  cat("Refs: Peng, Quesnelle, Sharkey & Chan (2023) 'YaRN: Efficient context\n")
  cat("      window extension of LLMs', arXiv:2309.00071; Chen et al (2023)\n")
  cat("      'Extending context window via positional interpolation', arXiv:2306.15595.\n")
}
