# Rainbow DQN (Reference Sec 47.145).
#
# Hessel et al 2018. DQN + Double + Duel + PER + n-step + Distributional
# + Noisy exploration. Deep-RL agent — R users typically call Python
# libraries (e.g. `reticulate::import("stable_baselines3")`) since no
# native R Rainbow implementation exists.

if (!requireNamespace("reticulate", quietly = TRUE)) install.packages("reticulate")
library(reticulate)                                   # bridge to Python

cat("=== Rainbow DQN (Hessel et al 2018) ===\n")
cat("No native R implementation of Rainbow DQN. Use one of:\n")
cat("  * stable_baselines3 (Python) via reticulate\n")
cat("  * cleanrl reference impls\n")
cat("  * ChainerRL / Ray RLlib\n")

# Toy tabular Double-Q-learning shell (illustrative)
n_states <- 7; n_actions <- 2
QA <- matrix(0, n_states, n_actions)
QB <- matrix(0, n_states, n_actions)
cat("\nInitialised Double-Q tables of size (7, 2) as illustration.\n")
