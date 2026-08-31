# Monte Carlo Tree Search (Reference §28.6)
# R via mcts or reticulate + Python.
# Run with:  Rscript monte_carlo_tree_search.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  mcts::create_node / do_search   -- basic MCTS with UCT\n")
  cat("Python:\n")
  cat("  open_spiel  -- DeepMind's game framework with MCTS baseline\n")
  cat("  alphazero-general -- SimonSchmidt implementation of AlphaZero-style MCTS + NN\n")
  cat("  mctspy, mcts (pypi) -- toy libraries\n")
  cat("  ray[rllib] AlphaZero / MuZero trainers\n")
  cat("Variants:\n")
  cat("  * UCT (Kocsis-Szepesvari 2006)  -- basic MCTS-UCB\n")
  cat("  * RAVE, MC-RAVE, all-moves-as-first\n")
  cat("  * AlphaGo/AlphaZero (Silver 2016/2017) -- policy + value network guided MCTS\n")
  cat("  * MuZero (Schrittwieser 2020) -- MCTS in a learned latent world model\n")
  cat("  * Progressive widening, PUCT, virtual loss (for parallel MCTS).\n")
}
