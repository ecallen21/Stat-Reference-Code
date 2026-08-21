# Language identification (Reference §25.10)
# R via cld2, cld3, textcat, or fastText.
# Run with:  Rscript language_detection.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  cld2::detect_language(x)          -- Google CLD2 (Naive Bayes on script + n-grams)\n")
  cat("  cld3::detect_language(x)          -- CLD3 neural language ID\n")
  cat("  textcat::textcat(x)               -- classical Cavnar-Trenkle n-gram profiles\n")
  cat("  fastTextR / lid.176.bin           -- FastText's high-accuracy 176-language model\n")
  cat("Python: langid, langdetect, fasttext.load_model('lid.176.bin').\n")
}
