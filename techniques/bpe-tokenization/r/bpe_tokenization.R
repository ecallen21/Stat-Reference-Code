# Byte-Pair Encoding (Reference Sec 47.159).
#
# Gage 1994; Sennrich et al 2016. Sub-word tokenization by merging
# most-frequent adjacent symbol pairs.
#
# Uses `tokenizers.bpe` for the SentencePiece / YouTokenToMe-style
# BPE, or `sentencepiece` for full SentencePiece.

if (!requireNamespace("tokenizers.bpe", quietly = TRUE)) install.packages("tokenizers.bpe")
library(tokenizers.bpe)                               # BPE tokenization

corpus <- rep(c("lower", "lowest", "newer", "newest", "widest", "wide",
                 "wider", "widening", "widened", "narrow", "narrower",
                 "narrowest", "narrowly"), 3)
tmp <- tempfile()
writeLines(corpus, tmp)
model <- bpe(tmp, vocab_size = 50, threads = 1)
cat("=== BPE tokenization (Gage 1994; Sennrich et al 2016) ===\n")
for (w in c("widen", "newest", "narrowly")) {
    cat(sprintf("  '%s' -> %s\n", w, toString(bpe_encode(model, w)[[1]])))
}
