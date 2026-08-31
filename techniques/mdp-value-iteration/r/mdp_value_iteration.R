# Value iteration + policy iteration on MDPs (Reference §28.2)
# R via MDPtoolbox or ReinforcementLearning.
# Run with:  Rscript mdp_value_iteration.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  MDPtoolbox::mdp_value_iteration(P, R, discount)\n")
  cat("  MDPtoolbox::mdp_policy_iteration(P, R, discount)\n")
  cat("  MDPtoolbox::mdp_LP(P, R, discount)         -- linear-programming solver\n")
  cat("  ReinforcementLearning::ReinforcementLearning(state, action, reward)\n")
  cat("Python:\n")
  cat("  scipy.optimize.linprog / cvxpy for LP formulation\n")
  cat("  gymnasium + numpy — from-scratch VI / PI on gridworld / cliff-walk\n")
  cat("Foundational references:\n")
  cat("  Bellman 1957, Howard 1960 policy iteration, Puterman 1994 MDPs textbook.\n")
}
