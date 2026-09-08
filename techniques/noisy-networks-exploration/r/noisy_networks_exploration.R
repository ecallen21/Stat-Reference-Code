# Noisy Networks for Exploration (Reference Sec 47.146).
#
# Fortunato et al 2018. Learnable parameter noise replaces
# epsilon-greedy exploration.
#
# No native R implementation; use Python via reticulate.

if (!requireNamespace("reticulate", quietly = TRUE)) install.packages("reticulate")
library(reticulate)                                   # bridge to Python

cat("=== NoisyNets exploration (Fortunato et al 2018) ===\n")
cat("No native R impl. Recommended libraries (Python):\n")
cat("  * cleanrl (dqn_atari_jax.py includes NoisyNet variant)\n")
cat("  * stable_baselines3 contrib\n")
cat("  * ChainerRL / Ray RLlib\n")

# Illustrative: sample a factorised-Gaussian noisy linear-layer weight
in_dim <- 4; out_dim <- 3; sigma <- 0.5
mu <- matrix(runif(in_dim * out_dim, -1 / sqrt(in_dim), 1 / sqrt(in_dim)),
              out_dim, in_dim)
f <- function(x) sign(x) * sqrt(abs(x))
e_in <- f(rnorm(in_dim)); e_out <- f(rnorm(out_dim))
noisy <- mu + (sigma / sqrt(in_dim)) * (e_out %*% t(e_in))
cat("Sampled noisy weight (row 1):", round(noisy[1, ], 3), "\n")
