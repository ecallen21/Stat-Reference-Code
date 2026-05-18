# Summary statistics for rates and proportions (Reference §1.8)
# From-scratch base-R plus idiomatic calls (binom.test, poisson.test, prop.test).
# Run with:  Rscript rates_proportions.R
#
# Inputs used below:
#   cases, population        : numerator and denominator for prevalence
#   new_cases, at_risk_start : for incidence proportion (risk)
#   follow_up_times          : numeric vector of per-subject follow-up durations
#   events, person_time_total: event count and total at-risk time, for incidence rate
#   x, n                     : successes / trials for proportion CIs
#   conf                     : confidence level (e.g. 0.95 -> 95% CI)

prevalence            <- function(cases, population) cases / population
incidence_proportion  <- function(new_cases, at_risk_start) new_cases / at_risk_start
person_time           <- function(follow_up_times) sum(follow_up_times)
incidence_rate        <- function(events, person_time_total) events / person_time_total

# --- CIs for a proportion p = x / n -------------------------------------
ci_wald <- function(x, n, conf = 0.95) {
  p <- x / n; z <- qnorm(0.5 + conf / 2); h <- z * sqrt(p * (1 - p) / n)
  c(max(0, p - h), min(1, p + h))
}
ci_wilson <- function(x, n, conf = 0.95) {
  p <- x / n; z <- qnorm(0.5 + conf / 2); z2 <- z^2
  denom <- 1 + z2 / n
  center <- (p + z2 / (2 * n)) / denom
  half <- (z / denom) * sqrt(p * (1 - p) / n + z2 / (4 * n^2))
  c(max(0, center - half), min(1, center + half))
}
ci_clopper_pearson <- function(x, n, conf = 0.95) {
  a <- 1 - conf
  lo <- if (x == 0) 0 else qbeta(a / 2, x, n - x + 1)
  hi <- if (x == n) 1 else qbeta(1 - a / 2, x + 1, n - x)
  c(lo, hi)
}

# --- exact Poisson CI for a rate ----------------------------------------
ci_poisson_rate <- function(events, person_time_total, conf = 0.95) {
  a <- 1 - conf
  lo_count <- if (events == 0) 0 else qchisq(a / 2, 2 * events) / 2
  hi_count <- qchisq(1 - a / 2, 2 * events + 2) / 2
  c(lo_count, hi_count) / person_time_total
}

# Library: stats::binom.test (Clopper-Pearson), stats::prop.test (Wilson),
#          stats::poisson.test (exact Poisson rate CI); epitools::pois.exact, binom::binom.confint
library_demo <- function(x, n, events, pt) {
  cat("binom.test (Clopper-Pearson):", binom.test(x, n)$conf.int, "\n")
  cat("prop.test (Wilson, correct=FALSE):", prop.test(x, n, correct = FALSE)$conf.int, "\n")
  cat("poisson.test rate CI:", poisson.test(events, pt)$conf.int, "\n")
}

if (sys.nframe() == 0) {
  cat("=== Prevalence ===\n120 / 5000 =", prevalence(120, 5000), "\n")
  follow <- c(2, 5, 1.5, 5, 3.2, 5, 0.8, 4.1); pt <- person_time(follow)
  cat("\n=== Incidence ===\nperson-time:", pt, "  rate (3 events):", incidence_rate(3, pt),
      "  incidence proportion (3/8):", incidence_proportion(3, 8), "\n")
  cat("\n=== 95% CIs for x=8, n=100 ===\n")
  cat("Wald           :", ci_wald(8, 100), "\n")
  cat("Wilson         :", ci_wilson(8, 100), "\n")
  cat("Clopper-Pearson:", ci_clopper_pearson(8, 100), "\n")
  cat("\n=== 95% exact Poisson CI: 3 events /", pt, "person-years ===\n")
  cat("rate =", 3 / pt, "  CI =", ci_poisson_rate(3, pt), "\n")
  cat("\n--- library ---\n"); library_demo(8, 100, 3, pt)
}
