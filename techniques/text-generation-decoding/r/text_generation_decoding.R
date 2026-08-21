# Decoding strategies for autoregressive text generation (Reference §25.x extra)
# R via torch or reticulate + huggingface transformers.
# Run with:  Rscript text_generation_decoding.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  torch: manual sampling loop on top of a causal-LM's per-step logits\n")
  cat("  reticulate + transformers pipeline('text-generation', ...) with kwargs:\n")
  cat("    do_sample=TRUE, temperature=..., top_k=..., top_p=..., num_beams=...\n")
  cat("Strategies:\n")
  cat("  * greedy                                     - argmax\n")
  cat("  * beam search (Bahdanau)                     - width B; deterministic + likelihood-max\n")
  cat("  * pure sampling / temperature (softmax scale) - controls entropy of the distribution\n")
  cat("  * top-k (Fan-Lewis-Dauphin 2018)             - restrict to k highest-p tokens\n")
  cat("  * top-p / nucleus (Holtzman 2019)             - smallest set with cumulative mass >= p\n")
  cat("  * typical decoding (Meister 2022), min-p, tail-free, contrastive, MBR.\n")
  cat("  * speculative decoding (Leviathan 2023)      - draft + verify for speed.\n")
}
