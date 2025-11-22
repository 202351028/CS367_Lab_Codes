import numpy as np
import matplotlib.pyplot as plt

class NonStationaryBandit:
    def __init__(self, num_arms=10, epsilon=0.1, steps=10000):
        self.num_arms = num_arms
        self.epsilon = epsilon
        self.steps = steps
        self.q_values = np.zeros(num_arms)
        self.action_counts = np.zeros(num_arms)
        self.true_means = np.zeros(num_arms)

        self.reward_history = []

    def drift_means(self):
        self.true_means += np.random.normal(0, 0.01, self.num_arms)

    def choose_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.num_arms)
        return np.argmax(self.q_values)

    def take_step(self):
        action = self.choose_action()

        reward = np.random.normal(self.true_means[action], 1)

        self.action_counts[action] += 1

        count = self.action_counts[action]
        self.q_values[action] += (reward - self.q_values[action]) / count

        self.drift_means()

        self.reward_history.append(reward)

    def simulate(self):
        for _ in range(self.steps):
            self.take_step()

        return self.q_values, self.reward_history


if __name__ == "__main__":
    bandit = NonStationaryBandit(num_arms=10, epsilon=0.1, steps=10000)

    final_q_vals, rewards = bandit.simulate()

    print("Estimated final Q-values:\n", final_q_vals)
    print("\nTotal reward collected:", np.sum(rewards))

    cumulative = np.cumsum(rewards)
    avg_reward = cumulative / (np.arange(1, len(rewards) + 1))

    plt.figure(figsize=(10,5))
    plt.plot(avg_reward, color='red', label='Average Reward')
    plt.xlabel('Steps')
    plt.ylabel('Average Reward')
    plt.title('Average Reward vs Steps (Non-stationary Bandit)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
