# Matrix factorisation ALS for RecSys (Reference Sec 47.118)
# Native R via recosystem; Python via implicit / LightFM.
# Run with:  Rscript matrix_factorization_als_recsys.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  recosystem                  -- LIBMF-based ALS / SGD for MF\n")
  cat("  softImpute                  -- nuclear-norm penalised matrix completion\n")
  cat("  rrecsys                     -- collaborative-filtering suite\n")
  cat("Python:\n")
  cat("  implicit                    -- Hu-Koren-Volinsky ALS + BPR + LMF\n")
  cat("  LightFM                     -- MF + side info via features\n")
  cat("  Surprise                    -- baseline SVD, KNN, SlopeOne\n")
  cat("  from-scratch                -- see matrix_factorization_als_recsys.py\n")
  cat("Refs: Hu, Koren & Volinsky (2008) 'Collaborative filtering for implicit\n")
  cat("      feedback datasets', ICDM; Koren, Bell & Volinsky (2009) Computer 42(8).\n")
}
