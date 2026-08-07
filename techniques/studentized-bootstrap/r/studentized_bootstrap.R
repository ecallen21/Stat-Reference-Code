# Studentized (bootstrap-t) confidence intervals (Reference §10.4)
# R via boot::boot with a statistic that returns (theta, var(theta)) then boot.ci(type = "stud").
# Run with:  Rscript studentized_bootstrap.R

if (sys.nframe() == 0) {
  set.seed(0); x <- rlnorm(40)
  if (requireNamespace("boot", quietly = TRUE)) {
    stat_mean <- function(data, idx) {
      d <- data[idx]
      c(mean(d), var(d) / length(d))
    }
    b <- boot::boot(x, stat_mean, R = 999)
    print(boot::boot.ci(b, type = c("basic", "perc", "stud")))
  }
}
