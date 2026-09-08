# Co-teaching for noisy labels (Reference Sec 47.98)
# Limited R support; Python via cleanlab / co-teaching.
# Run with:  Rscript noisy_label_cotraining.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  no established co-teaching in R -- roll your own with two learners\n")
  cat("  robustlmm / MASS::rlm       -- robust regression for outlier resistance\n")
  cat("Python:\n")
  cat("  cleanlab                    -- confident learning, noise-rate estimation\n")
  cat("  co-teaching (Bo Han)        -- author's PyTorch reference\n")
  cat("  DivideMix / JoCoR           -- successor methods\n")
  cat("  from-scratch                -- see noisy_label_cotraining.py\n")
  cat("Refs: Han et al (2018) 'Co-teaching', NeurIPS; Northcutt, Jiang & Chuang\n")
  cat("      (2021) 'Confident learning: Estimating uncertainty in dataset labels',\n")
  cat("      JAIR 70.\n")
}
