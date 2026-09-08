# Byte-Pair Encoding tokeniser (Reference Sec 47.83)
# Native R via text2vec / tokenizers; Python via HF tokenizers / subword-nmt.
# Run with:  Rscript byte_pair_encoding_bpe.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  text2vec::bpe                 -- BPE learner + tokeniser\n")
  cat("  tokenizers                    -- whitespace / n-gram tokenisers\n")
  cat("  sentencepiece                 -- SentencePiece bindings (BPE + Unigram-LM)\n")
  cat("Python:\n")
  cat("  tokenizers.models.BPE (HF)    -- trains + serialises industrial-scale BPE\n")
  cat("  subword-nmt (Sennrich)        -- reference BPE for NMT\n")
  cat("  sentencepiece                 -- SentencePiece BPE / Unigram\n")
  cat("  tiktoken                      -- OpenAI byte-level BPE tokenisers\n")
  cat("  from-scratch                  -- see byte_pair_encoding_bpe.py\n")
  cat("Refs: Sennrich, Haddow & Birch (2016) ACL; Gage (1994) C/C++ Users Journal;\n")
  cat("      Kudo & Richardson (2018) 'SentencePiece', EMNLP demo.\n")
}
