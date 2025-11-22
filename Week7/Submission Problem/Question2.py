import numpy as np
import matplotlib.pyplot as plt

class BinaryBandit:
    def __init__(self, p_success):
        self.ps = p_success

    def pull(self, action):
        if action not in (1, 2):
            raise ValueError("Action must be 1 or 2")
        return 1 if np.random.rand() < self.ps[action - 1] else 0


class EpsilonGreedyAgent:
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon
        self.counts = {1: 0, 2: 0}
        self.values = {1: 0.0, 2: 0.0}

    def select(self):
        if np.random.rand() < self.epsilon:
            return np.random.choice([1, 2])

        if self.values[1] == self.values[2]:
            return np.random.choice([1, 2])

        return 1 if self.values[1] > self.values[2] else 2

    def update(self, action, reward):
        self.counts[action] += 1
        n = self.counts[action]
        self.values[action] += (reward - self.values[action]) / n  


def run_binary_experiment(n_steps=1000, epsilon=0.1,
                          banditA_p=(0.1, 0.2),
                          banditB_p=(0.8, 0.9)):

    banditA = BinaryBandit(banditA_p)
    banditB = BinaryBandit(banditB_p)
    agent = EpsilonGreedyAgent(epsilon)

    cumulative_reward = 0
    avg_rewards = []
    instant_rewards = []

    for t in range(1, n_steps + 1):
        action = agent.select()

        rA = banditA.pull(action)
        rB = banditB.pull(action)
        reward = max(rA, rB)

        instant_rewards.append(reward)

        agent.update(action, reward)

        cumulative_reward += reward
        avg_rewards.append(cumulative_reward / t)

    print(f"\nAfter {n_steps} steps (epsilon={epsilon}):")
    print(f"Estimated values: Action1={agent.values[1]:.4f}, Action2={agent.values[2]:.4f}")
    print(f"Action counts: Action1={agent.counts[1]}, Action2={agent.counts[2]}")
    print(f"Final average reward: {avg_rewards[-1]:.4f}")

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(avg_rewards, color="blue")
    plt.title("Average Reward vs Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Average Reward")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(instant_rewards, color="green")
    plt.title("Reward vs Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Reward")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_binary_experiment(n_steps=2000, epsilon=0.1)
