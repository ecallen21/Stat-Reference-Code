# Markov transition models for longitudinal categorical data (Reference §12.9)
# Base R (matrix ops) + markovchain package for utilities.
# Run with:  Rscript markov_transition_models.R

fit_transition <- function(sequences, states = NULL) {
  if (is.null(states)) states <- sort(unique(unlist(sequences)))
  K <- length(states); idx <- setNames(seq_len(K), states)
  N <- matrix(0, K, K, dimnames = list(states, states))
  for (seq in sequences) for (i in seq_len(length(seq) - 1))
    N[idx[[seq[i]]], idx[[seq[i + 1]]]] <- N[idx[[seq[i]]], idx[[seq[i + 1]]]] + 1
  rt <- rowSums(N); P <- N / pmax(rt, 1)
  e <- eigen(t(P)); stat_idx <- which.min(abs(e$values - 1))
  stat <- Re(e$vectors[, stat_idx]); stat <- stat / sum(stat)
  list(states = states, counts = N, P = P,
       stationary = setNames(stat, states),
       n_transitions = sum(N))
}

if (sys.nframe() == 0) {
  set.seed(37); states <- c("healthy", "at_risk", "diagnosed")
  true_P <- rbind(c(0.85, 0.12, 0.03),
                  c(0.20, 0.60, 0.20),
                  c(0.05, 0.10, 0.85))
  simulate_one <- function(K = 6) {
    s <- "healthy"; out <- s
    for (i in 2:K) {
      probs <- true_P[which(states == s), ]
      s <- sample(states, 1, prob = probs); out <- c(out, s)
    }; out
  }
  sequences <- lapply(1:300, function(.) simulate_one())
  cat("=== MLE transition matrix ===\n")
  print(fit_transition(sequences, states))
  if (requireNamespace("markovchain", quietly = TRUE)) {
    cat("\n--- library: markovchain::markovchainFit ---\n")
    seqs_concat <- unlist(sequences)
    print(markovchain::markovchainFit(seqs_concat))
  }
}
