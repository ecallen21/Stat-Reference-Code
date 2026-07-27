# Weighted log-rank tests + stratified log-rank (Reference §11.4-§11.7, §11.47, §11.62)
# From-scratch base R + survival::survdiff as library cross-check.
# Run with:  Rscript log_rank_test.R

pooled_km_at <- function(times, events, event_times) {
  ord <- order(times); t <- times[ord]; e <- events[ord]
  sapply(event_times, function(tj) {
    S <- 1
    for (tk in event_times) {
      if (tk >= tj) break
      n_k <- sum(t >= tk); d_k <- sum((t == tk) & (e == 1))
      if (n_k > 0) S <- S * (1 - d_k / n_k)
    }
    S
  })
}

log_rank_test <- function(times, events, group,
                           weight_scheme = "logrank", rho = 0, gamma = 0) {
  labels <- sort(unique(group))
  if (length(labels) != 2) stop("need exactly 2 groups")
  a <- labels[1]
  event_times <- sort(unique(times[events == 1]))
  S_prev <- if (weight_scheme %in% c("peto", "fh"))
    pooled_km_at(times, events, event_times) else NULL
  U <- 0; V <- 0
  for (j in seq_along(event_times)) {
    tj <- event_times[j]
    n_j <- sum(times >= tj); d_j <- sum((times == tj) & (events == 1))
    n_Aj <- sum((times >= tj) & (group == a))
    d_Aj <- sum((times == tj) & (events == 1) & (group == a))
    if (n_j <= 1 || d_j == 0) next
    E_Aj <- n_Aj * d_j / n_j
    var_j <- n_Aj * (n_j - n_Aj) * d_j * (n_j - d_j) / (n_j^2 * (n_j - 1))
    w_j <- switch(weight_scheme,
      "logrank"     = 1,
      "wilcoxon"    = n_j,
      "peto"        = S_prev[j],
      "fh"          = S_prev[j]^rho * (1 - S_prev[j])^gamma,
      "tarone-ware" = sqrt(n_j))
    U <- U + w_j * (d_Aj - E_Aj); V <- V + w_j^2 * var_j
  }
  chi2 <- U^2 / V
  list(U = U, Var = V, chi_square = chi2, df = 1,
       p_value = pchisq(chi2, 1, lower.tail = FALSE),
       weight_scheme = weight_scheme)
}

stratified_log_rank <- function(times, events, group, strata,
                                 weight_scheme = "logrank") {
  Utot <- 0; Vtot <- 0
  for (s in unique(strata)) {
    m <- strata == s
    if (length(unique(group[m])) != 2) next
    r <- log_rank_test(times[m], events[m], group[m], weight_scheme)
    Utot <- Utot + r$U; Vtot <- Vtot + r$Var
  }
  chi2 <- Utot^2 / Vtot
  list(U_total = Utot, Var_total = Vtot, chi_square = chi2, df = 1,
       p_value = pchisq(chi2, 1, lower.tail = FALSE))
}

if (sys.nframe() == 0) {
  set.seed(7); n <- 100
  group <- sample(0:1, n, replace = TRUE)
  T_event <- rexp(n, rate = ifelse(group == 1, 0.3, 0.15))
  C_censor <- runif(n, 0, 10)
  times <- pmin(T_event, C_censor); events <- as.integer(T_event <= C_censor)
  for (s in c("logrank", "wilcoxon", "peto", "tarone-ware")) {
    r <- log_rank_test(times, events, group, s)
    cat("===", s, "===\n"); cat("chi2 =", round(r$chi_square, 4), ", p =", r$p_value, "\n")
  }
  if (requireNamespace("survival", quietly = TRUE)) {
    cat("\n--- library: survival::survdiff (logrank) ---\n")
    print(survival::survdiff(survival::Surv(times, events) ~ group))
  }
}
