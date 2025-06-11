from collections import defaultdict

from grid_world import GridWorld


def value_iter_onestep(V: defaultdict, env: GridWorld, gamma):
    for state in env.states():
        if state == env.goal_state:
            V[state] = 0
            continue

        action_values = []
        # if pi is once given by greedy policy, we can treat it as deterministic
        for action in env.actions():
            next_state = env.next_state(state, action)

            r = env.reward(state, action, next_state)
            value = r + gamma * V[next_state]
            action_values.append(value)

        # only the maximum value is used for the state because the probability for the best action is 1.0
        V[state] = max(action_values)

    return V


def value_iter(V: defaultdict, env: GridWorld, gamma, threshold=1e-3, is_render=True):
    # same is policy_iter
    while True:
        if is_render:
            env.render_v(V)

        old_V = V.copy()
        V = value_iter_onestep(V, env, gamma)

        delta = 0
        for state in V.keys():
            delta = max(delta, abs(old_V[state] - V[state]))

        if delta < threshold:
            break

    return V


if __name__ == "__main__":
    from policy_iter import greedy_policy

    V = defaultdict(lambda: 0.0)
    env = GridWorld()
    gamma = 0.9

    V = value_iter(V, env, gamma, is_render=False)

    pi = greedy_policy(V, env, gamma)
    env.render_v(V, pi)
