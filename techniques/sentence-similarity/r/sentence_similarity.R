# Semantic textual similarity (Reference §25.x extra)
# R via text, textEmbed, quanteda, or reticulate + sentence-transformers.
# Run with:  Rscript sentence_similarity.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  text::textEmbed(x, model='bert-base-uncased')       -- HuggingFace via reticulate\n")
  cat("  text2vec::sim2(vecs1, vecs2, method='cosine')       -- cosine on averaged word vectors\n")
  cat("  qdap / quanteda for tokenisation + bag-of-embedding averaging\n")
  cat("  textdata::embedding_glove6b + reticulate for pretrained GloVe\n")
  cat("Python: sentence-transformers ('all-MiniLM-L6-v2', 'gte-large', 'e5-large'),\n")
  cat("        openai / anthropic / voyage embedding APIs, spaCy nlp.similarity.\n")
  cat("Bench: STS-B, SICK, SemEval STS 2012-2017; report Pearson / Spearman correlation with humans.\n")
}
