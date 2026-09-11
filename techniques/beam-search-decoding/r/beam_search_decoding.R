# Beam Search Decoding (Reference Sec 47.158).
#
# Reddy 1977; Lowerre 1976. Approximate best-first sequence search.
#
# Pure-R implementation on a bigram transition model.

beam_search <- function(next_logits_fn, start_token, end_token,
                          max_len = 15, beam_width = 5, length_penalty = 0.7) {
    beams <- list(list(score = 0, seq = c(start_token), done = FALSE))
    for (step in seq_len(max_len)) {
        cands <- list()
        for (b in beams) {
            if (b$done) { cands[[length(cands) + 1]] <- b; next }
            lp <- next_logits_fn(b$seq)
            top <- order(lp, decreasing = TRUE)[seq_len(beam_width)]
            for (v in top) {
                new_score <- b$score + lp[v]
                new_seq <- c(b$seq, v)
                cands[[length(cands) + 1]] <- list(
                    score = new_score, seq = new_seq, done = (v == end_token))
            }
        }
        norm_scores <- sapply(cands, function(c) c$score / max(length(c$seq), 1)^length_penalty)
        keep <- order(norm_scores, decreasing = TRUE)[seq_len(beam_width)]
        beams <- cands[keep]
        if (all(sapply(beams, function(b) b$done))) break
    }
    beams
}

V <- 8; start <- 1; end <- 8
set.seed(0)
T_trans <- matrix(runif(V * V) + 0.1, V, V)
T_trans[1, ] <- c(0, 0.55, 0.10, 0.10, 0.15, 0.05, 0.03, 0.02)
T_trans[5, ] <- c(0.01, 0.01, 0.01, 0.01, 0.01, 0.90, 0.03, 0.02)
T_trans[6, ] <- c(0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.90, 0.04)
T_trans[7, ] <- c(0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.93)
T_trans <- T_trans / rowSums(T_trans)
nl <- function(seq) log(T_trans[seq[length(seq)], ] + 1e-12)

cat("=== Beam search (Reddy 1977; Lowerre 1976) ===\n")
for (B in c(1, 2, 5)) {
    b <- beam_search(nl, start, end, max_len = 15, beam_width = B)
    top <- b[[which.max(sapply(b, function(x) x$score))]]
    cat(sprintf("  Beam B = %d: seq = %s   log_prob = %.3f\n",
                 B, toString(top$seq), top$score))
}
