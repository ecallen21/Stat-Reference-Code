# String similarity (Reference §25.9)
# R via stringdist, RecordLinkage, or fuzzyjoin.
# Run with:  Rscript string_similarity.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  stringdist::stringdist(a, b, method='lv' | 'dl' | 'jw' | 'jaccard' | 'cosine' | 'soundex')\n")
  cat("  stringdist::stringsim (similarity in [0, 1])\n")
  cat("  RecordLinkage::compare.dedup / compare.linkage  -- record linkage helpers\n")
  cat("  fuzzyjoin::stringdist_inner_join(a, b, by='name', method='jw', max_dist=0.15)\n")
  cat("  phonics::soundex / metaphone / nysiis            -- phonetic hashing\n")
  cat("Python: rapidfuzz (fast Levenshtein + Jaro-Winkler + partial-ratio),\n")
  cat("        jellyfish (Soundex, Metaphone, Match-Rating).\n")
}
