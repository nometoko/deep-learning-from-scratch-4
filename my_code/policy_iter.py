from collections import defaultdict
from typing import Tuple

from policy_eval import policy_eval
from grid_world import GridWorld


def argmax(d: dict) -> int:
    return max(d, key=d.get)


def greedy_policy(
    V: defaultdict, env: GridWorld, gamma
) -> defaultdict[Tuple[int, int], dict[int, float]]:
    pi = {}
    for state in env.states():
        action_values = {}

        for action in env.actions():
            next_state = env.next_state(state, action)
            reward = env.reward(state, action, next_state)
            action_values[action] = reward + gamma * V[next_state]

        max_action = argmax(action_values)
        action_probs = {a: 0.0 for a in env.actions()}
        action_probs[max_action] = 1.0
        pi[state] = action_probs

    return pi


def policy_iter(env: GridWorld, gamma: float, threshold=1e-3, is_render=False):
    pi: defaultdict[Tuple[int, int], dict[int, float]] = defaultdict(
        lambda: {a: 1.0 / len(env.actions()) for a in env.actions()}
    )
    V = defaultdict(lambda: 0.0)

    while True:
        V = policy_eval(pi, V, env, gamma, threshold)
        new_pi = greedy_policy(V, env, gamma)

        if is_render:
            env.render_v(V, pi, print_value=True)

        if new_pi == pi:
            break
        pi = new_pi

    return pi


if __name__ == "__main__":
    env = GridWorld()
    gamma = 0.9
    pi = policy_iter(env, gamma, is_render=True)
