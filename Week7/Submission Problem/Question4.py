import numpy as np
import matplotlib.pyplot as plt

# Non-Stationary Bandit
class NonStationaryBandit:
    def __init__(self, num_arms=10, sigma_rw=0.01):
        self.num_arms = num_arms
        self.sigma_rw = sigma_rw
        self.true_means = np.zeros(num_arms)
        self.reward_history = []

    def random_walk(self):
        self.true_means += np.random.normal(0, self.sigma_rw, self.num_arms)

    def pull(self, action):
        reward = np.random.normal(self.true_means[action], 1.0)
        self.random_walk()
        self.reward_history.append(reward)
        return reward


# Epsilon-Greedy Agent
class EpsilonGreedyAgent:
    def __init__(self, num_arms=10, epsilon=0.1, alpha=0.1):
        self.num_arms = num_arms
        self.epsilon = epsilon
        self.alpha = alpha
        self.q_values = np.zeros(num_arms)
        self.action_counts = np.zeros(num_arms)

    def choose(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.num_arms)
        return np.argmax(self.q_values)

    def update(self, action, reward):
        self.q_values[action] += self.alpha * (reward - self.q_values[action])
        self.action_counts[action] += 1


# Simulation loop
def run_simulation(bandit, agent, steps=10000):
    rewards = np.zeros(steps)
    actions = np.zeros(steps)

    for t in range(steps):
        action = agent.choose()
        reward = bandit.pull(action)
        agent.update(action, reward)

        rewards[t] = reward
        actions[t] = action

    return rewards, actions


if __name__ == "__main__":
    np.random.seed(1)

    bandit = NonStationaryBandit(num_arms=10, sigma_rw=0.01)
    agent = EpsilonGreedyAgent(num_arms=10, epsilon=0.1, alpha=0.7)

    rewards, actions = run_simulation(bandit, agent, steps=10000)

    cumulative = np.cumsum(rewards)
    avg_rewards = cumulative / (np.arange(1, len(rewards) + 1))

    plt.figure(figsize=(10, 5))
    plt.plot(avg_rewards, label="Average Reward", color="blue")
    plt.title("Epsilon-Greedy Performance for Non-Stationary Bandit")
    plt.xlabel("Steps")
    plt.ylabel("Average Reward")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

