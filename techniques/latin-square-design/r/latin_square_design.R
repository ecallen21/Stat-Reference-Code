# Latin square design + ANOVA (Reference §16.6)
# Base R: aov(y ~ row + col + trt).
# Run with:  Rscript latin_square_design.R

if (sys.nframe() == 0) {
  set.seed(0); k <- 5
  ls <- outer(0:(k - 1), 0:(k - 1), function(i, j) (i + j) %% k)
  row_eff <- c(0, 0.5, -0.2, 0.3, -0.1)
  col_eff <- c(-0.1, 0.2, 0.4, -0.2, 0.1)
  trt_eff <- c(0, 0.5, 1, 1.5, 2)
  df <- expand.grid(row = 0:(k - 1), col = 0:(k - 1))
  df$trt <- as.vector(t(ls))
  df$y <- 10 + row_eff[df$row + 1] + col_eff[df$col + 1] +
          trt_eff[df$trt + 1] + rnorm(nrow(df), 0, 0.3)
  df$row <- factor(df$row); df$col <- factor(df$col); df$trt <- factor(df$trt)
  cat("=== Latin square ANOVA ===\n")
  print(summary(aov(y ~ row + col + trt, data = df)))
}
