import numpy as np
import matplotlib.pyplot as plt
import random

# Define DJCM Patterns (5×5, values -1 / 1)
D = np.array([-1,-1,-1,-1,1, 1,-1,1,1,-1, 1,-1,1,1,-1, 1,-1,1,1,-1, 1,-1,-1,-1,1]).reshape(5,5)
J = np.array([-1,-1,-1,-1,-1, 1,1,1,-1,1, 1,1,1,1,-1, 1,-1,1,1,-1, -1,-1,-1,1,1]).reshape(5,5)
C = np.array([1,-1,-1,-1,-1, -1,1,1,1,1, -1,1,1,1,1, -1,1,1,1,1, 1,-1,-1,-1,-1]).reshape(5,5)
M = np.array([-1,1,1,1,-1, -1,-1,1,-1,-1, -1,1,-1,1,-1, -1,1,1,1,-1, -1,1,1,1,-1]).reshape(5,5)

patterns = np.array([D, J, C, M])
names = ["D", "J", "C", "M"]

# Display stored patterns
plt.figure(figsize=(6,6))
for i, p in enumerate(patterns):
    plt.subplot(1, 4, i+1)
    plt.imshow(p, cmap='gray')
    plt.title(names[i])
    plt.axis("off")
plt.suptitle("Stored Patterns (D, J, C, M)")
plt.show()

# Hebbian Learning (bipolar Hopfield)
num = len(patterns)
dim = 25
W = np.zeros((dim, dim))

for p in patterns:
    v = p.reshape(-1)
    W += np.outer(v, v)

np.fill_diagonal(W, 0)
W = W / num  # normalization

# Function to introduce i bit errors
def corrupt(pattern, k):
    clean = pattern.copy()
    noisy = pattern.copy()
    flipped_locations = set()

    while len(flipped_locations) < k:
        r = np.random.randint(5)
        c = np.random.randint(5)
        if (r, c) not in flipped_locations:
            noisy[r,c] *= -1
            flipped_locations.add((r,c))

    return clean, noisy

# Iterative Hopfield update until convergence
def hopfield_recall(x):
    y = x.copy()
    prev_diff = None
    diff = 999

    while diff != prev_diff:
        prev_diff = diff
        y_new = np.sign(W @ y)
        diff = np.linalg.norm(y_new - y)
        y = y_new

    return y

# Plot: original, noisy, corrected (for 1 to 10 errors)
plt.figure(figsize=(10, 25))
plt.suptitle("Error Correction for Patterns D, J, C, M\n(1 to 10 Bit Flips)", fontsize=18)

for row in range(1, 11):

    # pick random pattern
    original = random.choice(patterns)

    clean, noisy = corrupt(original, row)
    corrected = hopfield_recall(noisy.reshape(-1)).reshape(5,5)

    # Original
    ax1 = plt.subplot(10, 3, 3*(row-1)+1)
    ax1.imshow(original, cmap="gray", vmin=-1, vmax=1)
    ax1.set_title("Original", fontsize=12)
    ax1.axis("off")

    # Noisy
    ax2 = plt.subplot(10, 3, 3*(row-1)+2)
    ax2.imshow(noisy, cmap="gray", vmin=-1, vmax=1)
    ax2.set_title(f"{row} Error(s)", fontsize=12)
    ax2.axis("off")

    # Corrected
    ax3 = plt.subplot(10, 3, 3*(row-1)+3)
    ax3.imshow(corrected, cmap="gray", vmin=-1, vmax=1)
    ax3.set_title("Corrected", fontsize=12)
    ax3.axis("off")

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()
