import numpy as np
import matplotlib.pyplot as plt


class Bandit:
    def __init__(self, n_arms=10):
        self.n_arms = n_arms
        self.rates = np.random.rand(n_arms)

    def play(self, arms):
        if arms < 0 or arms >= self.n_arms:
            raise ValueError("Invalid arm index")
        rate = self.rates[arms]

        if rate > np.random.rand():
            return 1
        else:
            return 0


class Agent:
    def __init__(self, epsilon, action_size=10):
        self.epsilon = epsilon
        self.Qs = np.zeros(action_size)
        self.ns = np.zeros(action_size)

    def update(self, action, reward):
        """
        Update the estimated value of the action based on the received reward.
        action: the action taken
        reward: the reward received from the action
        """
        self.ns[action] += 1
        self.Qs[action] += (reward - self.Qs[action]) / self.ns[action]

    def get_action(self):
        """
        Select an action based on the epsilon-greedy strategy.
        return: the selected action
        """
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, len(self.Qs))
        else:
            return np.argmax(self.Qs)


class AlphaAgent:
    def __init__(self, epsilon, alpha, action_size=10):
        self.epsilon = epsilon
        self.alpha = alpha
        self.Qs = np.zeros(action_size)

    def update(self, action, reward):
        """
        Update the estimated value of the action based on the received reward.
        action: the action taken
        reward: the reward received from the action
        """
        self.Qs[action] += (reward - self.Qs[action]) * self.alpha

    def get_action(self):
        """
        Select an action based on the epsilon-greedy strategy.
        return: the selected action
        """
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, len(self.Qs))
        else:
            return np.argmax(self.Qs)


def main(runs, steps, epsilon, n_arms=10):
    all_rates = np.zeros((runs, steps))

    for run in range(runs):
        bandit = Bandit(n_arms)
        agent = Agent(epsilon, n_arms)
        total_reward = 0
        rates = []

        for step in range(steps):
            action = agent.get_action()
            reward = bandit.play(action)
            agent.update(action, reward)
            total_reward += reward
            rates.append(total_reward / (step + 1))

        all_rates[run] = rates

    avg_rates = np.mean(all_rates, axis=0)

    plt.ylabel("Average reward")
    plt.xlabel("Steps")
    plt.plot(avg_rates, label="Average Reward")
    plt.show()


if __name__ == "__main__":
    main(200, 1000, 0.1, n_arms=10)
