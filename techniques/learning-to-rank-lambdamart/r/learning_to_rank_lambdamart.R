# LambdaMART LTR (Reference Sec 47.129)
# Native R via xgboost / lightgbm; Python via lightgbm / xgboost / TF Ranking.
# Run with:  Rscript learning_to_rank_lambdamart.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  xgboost with objective='rank:ndcg' / 'rank:pairwise'\n")
  cat("  lightgbm with objective='lambdarank'\n")
  cat("  ranger                          -- random forest ranker (weaker)\n")
  cat("Python:\n")
  cat("  lightgbm.LGBMRanker             -- reference LambdaMART\n")
  cat("  xgboost.XGBRanker               -- LambdaMART / MART\n")
  cat("  tensorflow_ranking              -- neural LTR (RankNet, DASALC, DLCM)\n")
  cat("  allrank / PyTerrier             -- research LTR libraries\n")
  cat("  from-scratch                    -- see learning_to_rank_lambdamart.py\n")
  cat("Refs: Burges (2010) 'From RankNet to LambdaRank to LambdaMART', MSR TR.\n")
}
