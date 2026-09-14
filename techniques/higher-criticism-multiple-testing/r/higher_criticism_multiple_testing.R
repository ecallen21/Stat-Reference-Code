# Higher Criticism (Donoho-Jin 2004 Ann Statist)
# R: `HHG::hhg.test` (loosely related detection), custom
#    implementation is standard; `harmonicmeanp` for HMP;
#    `dHSIC` for independence tests.
# Python: `statsmodels.stats.multitest` (BH, BY, Storey),
#         `mne.stats.permutation_cluster_test`,
#         from-scratch HC
#
# hc <- function(p) {
#   n <- length(p); p_sorted <- sort(p); i <- seq_len(n)
#   valid <- i / n >= 0.005 & i / n <= 0.5
#   hc_i <- sqrt(n) * (i / n - p_sorted) /
#           sqrt(p_sorted * (1 - p_sorted) + 1e-30)
#   max(hc_i[valid])
# }
