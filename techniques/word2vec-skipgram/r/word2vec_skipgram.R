# word2vec skip-gram (Reference Sec 47.117)
# Native R via text2vec / word2vec; Python via gensim.
# Run with:  Rscript word2vec_skipgram.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  text2vec::GlobalVectors     -- GloVe (matrix-factorisation cousin)\n")
  cat("  word2vec                    -- reference SG / CBOW in R\n")
  cat("  fastText                    -- subword-aware extension\n")
  cat("Python:\n")
  cat("  gensim.models.Word2Vec      -- reference SG / CBOW + negative sampling\n")
  cat("  gensim.models.FastText      -- subword-aware fastText\n")
  cat("  spacy / stanza              -- ship pretrained embeddings\n")
  cat("  from-scratch                -- see word2vec_skipgram.py\n")
  cat("Refs: Mikolov, Chen, Corrado & Dean (2013a) ICLR-W; Mikolov, Sutskever,\n")
  cat("      Chen, Corrado & Dean (2013b) NeurIPS.\n")
}
