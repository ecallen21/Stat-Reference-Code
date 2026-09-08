# Top-k / Top-p (nucleus) decoding (Reference Sec 47.110)
# LLM-only; Python via transformers.generation.
# Run with:  Rscript topk_topp_nucleus_sampling.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  no first-class decoding library in R -- write your own after logit output\n")
  cat("Python:\n")
  cat("  transformers.generation.LogitsProcessor + TopKLogitsWarper + TopPLogitsWarper\n")
  cat("  transformers.GenerationConfig(top_k=..., top_p=..., temperature=..., do_sample=True)\n")
  cat("  vllm.SamplingParams(top_k, top_p, temperature)\n")
  cat("  litellm / openai         -- API-level sampling knobs\n")
  cat("  from-scratch             -- see topk_topp_nucleus_sampling.py\n")
  cat("Refs: Fan, Lewis & Dauphin (2018) 'Hierarchical neural story generation', ACL;\n")
  cat("      Holtzman, Buys, Du, Forbes & Choi (2020) ICLR.\n")
}
