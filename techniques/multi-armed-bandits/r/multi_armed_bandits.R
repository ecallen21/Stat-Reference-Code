# Multi-armed bandits (Reference §28.1)
# R via contextual, bandit, or roll-your-own.
# Run with:  Rscript multi_armed_bandits.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  contextual::Simulator + Agent + Policy (EpsilonGreedy, UCB1, Thompson, LinUCB, ...)\n")
  cat("  bandit::bandit (base R implementation)\n")
  cat("  Python: mabwiser, contextualbandits (Alberto Miranda), vowpalwabbit --cb\n")
  cat("Family:\n")
  cat("  * epsilon-greedy / eps-decreasing / softmax\n")
  cat("  * UCB1, UCB-tuned, UCB-V\n")
  cat("  * Thompson sampling (Beta for Bernoulli; Gaussian; Bayesian LR for contextual)\n")
  cat("  * LinUCB (Li 2010) / disjoint / hybrid — contextual linear bandits\n")
  cat("  * Neural bandits — deep contextual with epsilon-greedy or Thompson via dropout\n")
  cat("Applications: A/B testing, recommender exploration, clinical adaptive trials.\n")
}
