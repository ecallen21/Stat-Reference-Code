# Monte Carlo Tree Search — MCTS (Reference §28.6)

Kocsis & Szepesvári (2006). Best-first search over a game / MDP tree,
guided by UCB1-style exploration.

## Four-phase rollout

Repeat `n_rollouts` times:

1. **Selection**: from the root, descend via UCT until reaching a node with untried actions or a terminal state.
   ```
   UCT(child) = Q(child) + c · √(ln N(parent) / N(child))
   ```
2. **Expansion**: pop one untried action, add the resulting child.
3. **Simulation** (rollout / playout): from that child, follow a random policy to a terminal state.
4. **Backpropagation**: propagate the terminal reward up through the visited nodes, updating `N` and `W`.

Root recommendation: usually the **most-visited** child (robust to noise),
or the highest-Q child.

## Family

| Variant | Key idea |
|---|---|
| **UCT** (Kocsis-Szepesvári 2006) | classic MCTS with UCB1 selection |
| **RAVE / AMAF** | share statistics across sibling subtrees |
| **AlphaGo / AlphaZero** (Silver 2016/2017) | replace random rollout with a **value network**; policy network provides prior for **PUCT** |
| **MuZero** (Schrittwieser 2020) | MCTS in a **learned** latent world model — no simulator needed |
| **Progressive widening** | grow branching factor with visit count; useful for continuous / stochastic actions |
| **Virtual loss + tree parallelisation** | multi-threaded MCTS |

## When to use

- **Board / card games** — Go, chess, Hex, poker; anywhere the game tree can be simulated.
- **Planning under a known model** — MPC, robotics with a known dynamics simulator, warehouse routing.
- **Combinatorial optimisation** — planning inside a tree of decisions with a rollout evaluator.
- **LLM reasoning** — recent work uses MCTS over reasoning steps (Tree-of-Thoughts, Q*, MCTSr).
- **Not** for continuous state without a simulator; use policy-gradient RL.

## Files

- `python/monte_carlo_tree_search.py` — from-scratch MCTS-UCT on a toy depth-3 binary tree where the sole winning path is `RRR` (reward +1) and every other leaf gives −1. After 500 rollouts:
  - Root children visits: R = 497, L = 3.
  - Root avg-Q: R = +0.97, L = −1.0.
  - Recommended action: R. Correct.
- `r/monte_carlo_tree_search.R` — `mcts` R package; Python `open_spiel`, `alphazero-general`, `ray[rllib]` AlphaZero / MuZero trainers.

## Assumptions & caveats

- **Exploration constant `c`** — √2 for reward-scale-normalised games; tune per domain.
- **Simulator required** — pure MCTS needs a forward model; MuZero learns it.
- **Rollout policy matters** — random is a weak baseline; a hand-crafted or learned rollout policy dramatically improves quality (as in AlphaGo Zero).
- **Progressive widening** for large / continuous action spaces.
- **Determinism vs stochasticity** — for stochastic environments, use **chance nodes** and separate exploration for those.
- **Rollout length / depth** — episodes must be short enough to terminate; use a value network at cut-off for AlphaZero-style MCTS.
- **Parallelism** — virtual loss + lock-free updates; tree grows correctly across threads.

## Related in this repo

- `reinforcement-learning-basics`, `mdp-value-iteration`, `dqn-deep-q-network` — RL alternatives.
- `multi-armed-bandits` — UCT reduces to UCB1 at each internal node.
- `bayesian-optimization` — a Bayesian planning cousin.
- `hmm`, `markov-transition-models` — transition-model neighbours.

## Run

```
python techniques/monte-carlo-tree-search/python/monte_carlo_tree_search.py
Rscript techniques/monte-carlo-tree-search/r/monte_carlo_tree_search.R
```

**Refs:** Kocsis, L. & Szepesvári, C. "Bandit based Monte-Carlo planning." *ECML*, 2006; Browne, C.B. et al. "A survey of Monte Carlo tree search methods." *IEEE Trans. Comput. Intell. AI Games* 4(1), 1–43, 2012; Silver, D. et al. "Mastering the game of Go without human knowledge (AlphaGo Zero)." *Nature* 550, 354–359, 2017; Schrittwieser, J. et al. "Mastering Atari, Go, chess and shogi by planning with a learned model (MuZero)." *Nature* 588, 604–609, 2020.

---

## Author

Elisabeth F. Callen, Ph.D., PStat®
Biostatistician and applied health data researcher

[LinkedIn](https://www.linkedin.com/in/your-profile) · [ORCID](https://orcid.org/your-id) · elisabeth.f.callen@gmail.com

## Acknowledgments

**AI tooling.** This codebase was developed with the support of AI coding assistants (Claude Code). Methodology, statistical approach, validation logic, and interpretation of results are my own. AI tooling was used to accelerate code drafting, refactor for readability, and assist with documentation. All code was reviewed, tested, and validated against expected outputs before committing.

No protected health information was ever provided to AI coding assistants. All development and testing was conducted against synthetic data.

## License

[MIT](../../LICENSE)
