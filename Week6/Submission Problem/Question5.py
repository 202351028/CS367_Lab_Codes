import numpy as np
import matplotlib.pyplot as plt

# Generate 10 random cities with coordinates
np.random.seed(0)
cities = [str(i) for i in range(1, 11)]
coord = {c: np.random.rand(2) * 50 for c in cities}

# Build distance matrix
def get_distances(cities, coords):
    n = len(cities)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                D[i, j] = np.linalg.norm(coords[cities[i]] - coords[cities[j]])
    return D

dist_matrix = get_distances(cities, coord)

# TSP Solver (Random Swap Descent)
class SimpleTSP:
    def __init__(self, distances):
        self.D = distances
        self.n = distances.shape[0]

    def route_length(self, route):
        total = 0
        for i in range(self.n - 1):
            total += self.D[route[i], route[i+1]]
        total += self.D[route[-1], route[0]]   # Return to start
        return total

    def optimize(self, iterations=120000):
        best_route = np.random.permutation(self.n)
        best_cost = self.route_length(best_route)

        for _ in range(iterations):
            # Propose swap
            a, b = np.random.choice(self.n, 2, replace=False)
            new_route = best_route.copy()
            new_route[a], new_route[b] = new_route[b], new_route[a]

            new_cost = self.route_length(new_route)

            if new_cost < best_cost:
                best_cost = new_cost
                best_route = new_route

        return best_route, best_cost

# Run solver
solver = SimpleTSP(dist_matrix)
best_route, best_cost = solver.optimize()

print("Optimal Tour:")
for i, idx in enumerate(best_route):
    if i == 0:
        print(f"Start from {cities[idx]}")
    else:
        print(f"Go to {cities[idx]}")
print(f"Return to {cities[best_route[0]]}")
print("Minimum Path Cost:", best_cost)

plt.figure(figsize=(10, 8))

for c in cities:
    plt.scatter(coord[c][0], coord[c][1], color='green')
    plt.text(coord[c][0], coord[c][1], c, fontsize=12, ha='center', va='center')

for i in range(len(best_route) - 1):
    c1 = cities[best_route[i]]
    c2 = cities[best_route[i+1]]
    plt.plot([coord[c1][0], coord[c2][0]],
             [coord[c1][1], coord[c2][1]], color='blue')

start_c = cities[best_route[0]]
end_c = cities[best_route[-1]]

plt.plot([coord[start_c][0], coord[end_c][0]],
         [coord[start_c][1], coord[end_c][1]], color='blue')

# Annotate distances on edges
for i in range(len(best_route) - 1):
    a = best_route[i]
    b = best_route[i+1]
    x1, y1 = coord[cities[a]]
    x2, y2 = coord[cities[b]]
    midx, midy = (x1 + x2) / 2, (y1 + y2) / 2
    plt.text(midx, midy, f"{dist_matrix[a,b]:.2f}", color="red")

plt.title("Traveling Salesman Problem — Optimized Tour")
plt.xlabel("X coordinate")
plt.ylabel("Y coordinate")
plt.grid(True)
plt.show()
