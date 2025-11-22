import numpy as np
import matplotlib.pyplot as plt

# Hopfield Network Class
class Hopfield:
    def __init__(self, size=100):
        self.N = size
        self.W = np.zeros((self.N, self.N))

    def train(self, patterns):
        self.W = np.zeros((self.N, self.N))
        for p in patterns:
            s = 2 * p - 1
            self.W += np.outer(s, s)
        np.fill_diagonal(self.W, 0)

    def recall(self, pattern, steps=10):
        s = 2 * pattern - 1
        for _ in range(steps):
            for i in np.random.permutation(self.N):
                h = np.dot(self.W[i], s)
                s[i] = 1 if h > 0 else -1
        return (s + 1) // 2

    def energy(self, state):
        s = 2 * state - 1
        return -0.5 * s @ (self.W @ s)

# Create 10x10 Hopfield Network
N = 100
model = Hopfield(N)

# generate random patterns
num_patterns = 12
patterns = [np.random.randint(0, 2, N) for _ in range(num_patterns)]

# train model
model.train(patterns)

# Test recall with noise
test = patterns[0].copy()
flip_count = 20
idx = np.random.choice(N, flip_count, replace=False)
test[idx] = 1 - test[idx]

recalled = model.recall(test, steps=8)

plt.figure(figsize=(10, 4))

titles = ["Original Pattern", "Noisy Pattern", "Recalled Pattern"]
images = [patterns[0], test, recalled]

for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.imshow(images[i].reshape(10, 10), cmap="gray")
    plt.title(titles[i])
    plt.axis("off")

plt.suptitle("Hopfield Network (10×10 Associative Memory)")
plt.tight_layout()
plt.show()

# Capacity Test
correct = 0
for p in patterns:
    out = model.recall(p)
    if np.array_equal(out, p):
        correct += 1

print(f"\nStored Patterns: {num_patterns}")
print(f"Correctly Recalled: {correct}/{num_patterns}")
print(f"Empirical Capacity ≈ {correct} patterns\n")
