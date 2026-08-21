# Text classification (Reference §25.6)
# R via quanteda.textmodels, e1071, or glmnet.
# Run with:  Rscript text_classification.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  quanteda.textmodels::textmodel_nb(dfm, y)               -- multinomial NB\n")
  cat("  quanteda.textmodels::textmodel_svm(dfm, y)              -- linear SVM on DFM\n")
  cat("  quanteda.textmodels::textmodel_wordscores(dfm, y)       -- Laver-Benoit scaling\n")
  cat("  glmnet::cv.glmnet(dtm, y, family='multinomial')          -- L1/L2 logistic on TF-IDF\n")
  cat("  text2vec::vectorizer + glmnet workflow                   -- pipeline\n")
  cat("  Python: sklearn.naive_bayes.MultinomialNB / ComplementNB,\n")
  cat("          sklearn.linear_model.LogisticRegression / SGDClassifier(loss='log_loss'),\n")
  cat("          fasttext, huggingface transformers.\n")
}
