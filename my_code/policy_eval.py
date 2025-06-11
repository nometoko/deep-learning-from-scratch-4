from collections import defaultdict
from typing import Tuple

from grid_world import GridWorld


def eval_onestep(pi: defaultdict, V: defaultdict, env: GridWorld, gamma: float = 0.9):
    for state in env.states():
        if state == env.goal_state:
            V[state] = 0
            continue

        action_probs = pi[state]
        new_V = 0

        for action, action_prob in action_probs.items():
            next_state = env.next_state(state, action)
            reward = env.reward(state, action, next_state)
            new_V += action_prob * (reward + gamma * V[next_state])

        V[state] = new_V

    return V


def policy_eval(
    pi: defaultdict[Tuple[int, int], dict[int, float]],
    V: defaultdict,
    env: GridWorld,
    gamma: float = 0.9,
    threshold: float = 1e-3,
):
    while True:
        old_V = V.copy()
        V = eval_onestep(pi, V, env, gamma)

        delta = 0

        for state in V.keys():
            delta = max(delta, abs(V[state] - old_V[state]))

        if delta < threshold:
            break

    return V


if __name__ == "__main__":
    from grid_world import GridWorld
    from collections import defaultdict

    env = GridWorld()
    gamma = 0.9
    pi = defaultdict(lambda: {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25})

    V = defaultdict(lambda: 0)

    V = policy_eval(pi, V, env, gamma)

    env.render_v(V, pi, print_value=True)
