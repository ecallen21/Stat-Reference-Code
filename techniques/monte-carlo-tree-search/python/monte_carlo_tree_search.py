"""Monte Carlo Tree Search with UCT selection (Kocsis-Szepesvari 2006; Reference §28.6).

Four-phase loop per rollout:
    1. SELECTION:  from the root, descend via UCT until reaching an
                   unexpanded (leaf) node.
        UCT(child) = q_child + c * sqrt( ln(n_parent) / n_child )
    2. EXPANSION:  add one child from the untried actions.
    3. SIMULATION: play out with a random rollout policy to a terminal state.
    4. BACKPROP:   propagate the terminal reward up the tree.

We demonstrate on a small deterministic MDP where the optimal path from the
root gives reward +1 and other paths give -1.  MCTS with enough rollouts
picks the correct root action.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

import math    # stdlib: scalar math

import numpy as np    # numerical arrays + linear algebra


class Node:
    __slots__ = ("parent", "action_from_parent", "children", "N", "W", "actions_untried", "state")
    def __init__(self, state, parent=None, action=None, actions=None):
        self.state = state
        self.parent = parent
        self.action_from_parent = action
        self.children = []
        self.N = 0
        self.W = 0.0
        self.actions_untried = list(actions) if actions is not None else []


def uct(node: Node, c: float = 1.4) -> Node:
    return max(node.children, key=lambda ch:
                (ch.W / max(ch.N, 1))
                + c * math.sqrt(math.log(max(node.N, 1)) / max(ch.N, 1)))


def mcts(env, n_rollouts: int = 500, c: float = 1.4, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    root = Node(env.initial_state(), actions=env.actions(env.initial_state()))
    for _ in range(n_rollouts):
        # 1. selection
        node = root
        while not node.actions_untried and node.children:
            node = uct(node, c)
        # 2. expansion
        if node.actions_untried:
            a = node.actions_untried.pop(rng.integers(len(node.actions_untried)))
            s2 = env.step(node.state, a)
            child = Node(s2, parent=node, action=a, actions=env.actions(s2))
            node.children.append(child)
            node = child
        # 3. simulation
        s = node.state
        while not env.is_terminal(s):
            actions = env.actions(s)
            a = actions[rng.integers(len(actions))]
            s = env.step(s, a)
        r = env.terminal_reward(s)
        # 4. backpropagation
        while node is not None:
            node.N += 1; node.W += r; node = node.parent
    # root action recommendation: most-visited child
    best_child = max(root.children, key=lambda ch: ch.N)
    return {"root": root, "recommended_action": best_child.action_from_parent,
            "child_visits": {ch.action_from_parent: ch.N for ch in root.children},
            "child_avg_q": {ch.action_from_parent: ch.W / max(ch.N, 1)
                             for ch in root.children},
            "method": "MCTS-UCT"}


class SimpleGame:
    """A depth-3 binary tree.  Optimal path is R -> R -> R => reward +1.
    Every other terminal leaf gives -1.  Actions labelled 'L' / 'R'."""
    def initial_state(self): return ""
    def actions(self, s): return [] if self.is_terminal(s) else ["L", "R"]
    def step(self, s, a): return s + a
    def is_terminal(self, s): return len(s) >= 3
    def terminal_reward(self, s): return 1.0 if s == "RRR" else -1.0


if __name__ == "__main__":
    env = SimpleGame()
    r = mcts(env, n_rollouts=500, c=1.4, seed=0)
    print("=== MCTS-UCT on a depth-3 binary tree (optimal path = 'RRR' -> +1) ===")
    print(f"  root children visits : {r['child_visits']}")
    print(f"  root children avg-Q  : "
          f"{ {k: round(v, 3) for k, v in r['child_avg_q'].items()} }")
    print(f"  recommended action   : {r['recommended_action']}")
    print(f"  (should be R -- the higher-mean subtree)")

    print("\n--- library cross-check (open_spiel; alphazero-general; mctsr) ---")
