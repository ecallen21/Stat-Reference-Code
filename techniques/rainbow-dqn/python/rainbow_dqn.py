"""Rainbow DQN (Reference Sec 47.145).

Hessel et al 2018 'Rainbow: Combining Improvements in Deep
Reinforcement Learning', AAAI. Combines six extensions of DQN:

    1. Double DQN         (van Hasselt 2016)  - decouple selection / evaluation
    2. Prioritized replay (Schaul 2016)       - sample high-TD-error transitions
    3. Dueling networks   (Wang 2016)         - separate V(s) and A(s, a) heads
    4. Multi-step returns (Sutton 1988)       - n-step TD targets
    5. Distributional RL  (Bellemare 2017)    - predict return distribution
    6. Noisy nets         (Fortunato 2018)    - exploration via noisy weights

Below is a minimal illustration on a small chain-MDP: tabular
Double-DQN (i.e. Double-Q-learning) + prioritized replay + n-step
returns. Rainbow's other three extensions require deep networks.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def chain_step(s, a, n_states=7):
    """Chain-MDP: left/right actions, reward +1 at right end, -0.01 elsewhere."""
    ns = np.clip(s + (1 if a == 1 else -1), 0, n_states - 1)
    r = 1.0 if ns == n_states - 1 else -0.01
    done = ns == n_states - 1
    return ns, r, done


def double_q_learning(n_states=7, n_episodes=400, alpha=0.5, gamma=0.95,
                        eps=0.1, n_step=3, per_alpha=0.6, seed=0):
    """Tabular Double-Q + n-step returns + prioritized replay."""
    rng = np.random.default_rng(seed)
    QA = np.zeros((n_states, 2))
    QB = np.zeros((n_states, 2))
    buffer, priorities = [], []
    returns_hist = []
    for ep in range(n_episodes):
        s = 0
        ep_ret = 0.0
        traj = []
        for t in range(50):
            if rng.uniform() < eps:
                a = int(rng.integers(0, 2))
            else:
                a = int(np.argmax(QA[s] + QB[s]))
            ns, r, done = chain_step(s, a, n_states)
            traj.append((s, a, r, ns, done))
            s = ns
            ep_ret += r
            if done: break
        # Compute n-step returns and enqueue
        for i in range(len(traj)):
            G = 0.0; disc = 1.0
            end_ns, end_done = traj[i][3], traj[i][4]
            for k in range(i, min(i + n_step, len(traj))):
                G += disc * traj[k][2]
                disc *= gamma
                end_ns, end_done = traj[k][3], traj[k][4]
                if end_done: break
            transition = (traj[i][0], traj[i][1], G, end_ns, end_done, disc)
            buffer.append(transition)
            priorities.append(1.0)                            # new -> high prio
        # PER sampling weighted by priority^alpha
        if len(buffer) > 32:
            p = np.array(priorities) ** per_alpha
            p /= p.sum()
            idx = rng.choice(len(buffer), size=32, p=p)
            for i in idx:
                s_i, a_i, G_i, ns_i, done_i, disc_i = buffer[i]
                if rng.uniform() < 0.5:                       # Double-Q update A
                    a_star = int(np.argmax(QA[ns_i]))
                    target = G_i + (0 if done_i else disc_i * QB[ns_i, a_star])
                    td = target - QA[s_i, a_i]
                    QA[s_i, a_i] += alpha * td
                else:                                          # update B
                    a_star = int(np.argmax(QB[ns_i]))
                    target = G_i + (0 if done_i else disc_i * QA[ns_i, a_star])
                    td = target - QB[s_i, a_i]
                    QB[s_i, a_i] += alpha * td
                priorities[i] = abs(td) + 0.01                # update priority
        returns_hist.append(ep_ret)
    Q = (QA + QB) / 2
    return {"Q": Q, "returns": returns_hist}


if __name__ == "__main__":
    print("=== Rainbow-DQN components (Hessel et al 2018) ===\n")
    r = double_q_learning(n_states=7, n_episodes=400, n_step=3, seed=0)
    Q = r["Q"]
    print("  Learned Q(s, a) after 400 eps (7-state chain, goal at right):")
    print(f"    state:  {list(range(7))}")
    print(f"    Q(left):  {np.round(Q[:, 0], 2)}")
    print(f"    Q(right): {np.round(Q[:, 1], 2)}")
    pol = np.argmax(Q, axis=1)
    print(f"    greedy policy (0=L, 1=R): {list(pol)}   (all-1 is optimal)")

    for n_step in [1, 3, 5]:
        r = double_q_learning(n_step=n_step, n_episodes=200, seed=0)
        # Average return of last 50 episodes
        late = float(np.mean(r["returns"][-50:]))
        print(f"\n  n-step = {n_step}   avg return (last 50 eps) = {late:.3f}")

    print("\n--- library cross-check (rlberry / cleanrl / stable-baselines3 Python) ---")
