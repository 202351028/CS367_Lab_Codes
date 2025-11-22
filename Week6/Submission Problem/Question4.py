import numpy as np
import matplotlib.pyplot as plt

# Create a random board with exactly 8 rooks
def random_rook_board(n=8):
    board = np.zeros((n, n), dtype=int)
    choices = np.random.permutation(n * n)[:n]
    for pos in choices:
        r, c = divmod(pos, n)
        board[r, c] = 1
    return board

# Penalize rows/columns having != 1 rook
def rook_energy(B):
    rows = B.sum(axis=1) - 1
    cols = B.sum(axis=0) - 1
    return np.sum(rows**2) + np.sum(cols**2)

# Try swapping a rook to an empty position
def minimize_energy(board, steps=1500):
    n = board.shape[0]
    E = rook_energy(board)

    for _ in range(steps):
        # pick two random squares
        p1, p2 = np.random.choice(n*n, 2, replace=False)
        r1, c1 = divmod(p1, n)
        r2, c2 = divmod(p2, n)

        # only consider valid swap: rook → empty
        if board[r1, c1] == 1 and board[r2, c2] == 0:
            # swap
            board[r1, c1], board[r2, c2] = 0, 1
            newE = rook_energy(board)

            # accept only if energy improves
            if newE < E:
                E = newE
            else:
                # revert
                board[r1, c1], board[r2, c2] = 1, 0

        # if perfect solution found
        if E == 0:
            break

    return board, E

# Initial random board
initial = random_rook_board()

plt.figure(figsize=(5,5))
plt.imshow(initial, cmap="binary")
plt.title("Initial Board (Rooks = 1)")
plt.axis("off")
plt.show()

# Optimize
solution, energy_final = minimize_energy(initial.copy())

print("Final Energy:", energy_final)

plt.figure(figsize=(5,5))
plt.imshow(solution, cmap="binary")
plt.title("Optimized Board (Eight Rooks)")
plt.axis("off")
plt.show()
