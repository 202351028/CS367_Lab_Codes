import numpy as np
import matplotlib.pyplot as plt
import random
import math
from typing import List, Tuple


class JigsawSolverSA:

    def __init__(self, file_path: str, grid_size: int = 4):
        # Configuration
        self.grid_size = grid_size
        self.total_tiles = grid_size * grid_size

        # Load and segment data
        self.raw_data = self._load_image_data(file_path)
        self.patches = self._split_into_patches()
        self.patch_dim = self.patches[0].shape[0]

        self.current_state = list(range(self.total_tiles))
        random.shuffle(self.current_state)  # Start with a random initial state


    def _load_image_data(self, file_path: str) -> np.ndarray:
        try:
            with open(file_path, "r") as f:
                # Skip the first 5 header lines based on the file format
                intensity_lines = f.readlines()[5:]

                # Convert valid lines to integers
                pixel_values = [int(line.strip()) for line in intensity_lines if line.strip()]

            # The expected size for a 4x4 grid reconstruction of Lena is 512x512
            if len(pixel_values) != 512 * 512:
                raise ValueError(f"Expected {512 * 512} pixels, but found {len(pixel_values)}. Check file format.")

            # Reshape to 512x512 and transpose
            return np.array(pixel_values, dtype=np.uint8).reshape((512, 512)).T

        except FileNotFoundError:
            print(f"Error: File not found at {file_path}")
            return np.zeros((512, 512))  # Return a zero array on failure
        except ValueError as e:
            print(f"Error processing image data: {e}")
            return np.zeros((512, 512))

    def _split_into_patches(self) -> List[np.ndarray]:
        patch_dim = self.raw_data.shape[0] // self.grid_size
        patches = []
        for r in range(0, self.raw_data.shape[0], patch_dim):
            for c in range(0, self.raw_data.shape[1], patch_dim):
                patch = self.raw_data[r:r + patch_dim, c:c + patch_dim]
                patches.append(patch)
        return patches

    def _calculate_edge_mismatch(self, patch_a: np.ndarray, patch_b: np.ndarray, direction: str) -> float:
        if direction == 'horizontal':
            # Right edge of A vs Left edge of B
            return np.sum(np.abs(patch_a[:, -1].astype(np.int16) - patch_b[:, 0].astype(np.int16)))
        elif direction == 'vertical':
            # Bottom edge of A vs Top edge of B
            return np.sum(np.abs(patch_a[-1, :].astype(np.int16) - patch_b[0, :].astype(np.int16)))
        else:
            return 0.0

    def calculate_total_cost(self, state: List[int]) -> float:
        total_cost = 0.0

        # Helper to get the patch index in the 4x4 grid
        get_patch_idx = lambda r, c: state[r * self.grid_size + c]

        for r in range(self.grid_size):
            for c in range(self.grid_size):
                patch_idx = get_patch_idx(r, c)
                patch_a = self.patches[patch_idx]

                # 1. Horizontal Mismatch (with right neighbor)
                if c < self.grid_size - 1:
                    patch_b_idx = get_patch_idx(r, c + 1)
                    patch_b = self.patches[patch_b_idx]
                    total_cost += self._calculate_edge_mismatch(patch_a, patch_b, 'horizontal')

                # 2. Vertical Mismatch (with bottom neighbor)
                if r < self.grid_size - 1:
                    patch_b_idx = get_patch_idx(r + 1, c)
                    patch_b = self.patches[patch_b_idx]
                    total_cost += self._calculate_edge_mismatch(patch_a, patch_b, 'vertical')

        return total_cost

    def _get_neighbor_state(self, current_state: List[int]) -> List[int]:
        """Generates a neighboring state by swapping two random tiles."""
        new_state = list(current_state)
        # Select two distinct random positions (not tile indices, but grid positions 0-15)
        idx1, idx2 = random.sample(range(self.total_tiles), 2)
        # Swap the tiles at these positions
        new_state[idx1], new_state[idx2] = new_state[idx2], new_state[idx1]
        return new_state

    def simulated_annealing(self,
                            max_iterations: int = 1_000_000,
                            initial_temp: float = 1.0,
                            cooling_rate: float = 0.99999) -> List[int]:
        print(f"Starting SA with T={initial_temp}, Iterations={max_iterations}, Cooling={cooling_rate}")

        current_state = self.current_state
        current_cost = self.calculate_total_cost(current_state)
        best_state = current_state
        best_cost = current_cost

        T = initial_temp
        i = 0  # Iteration counter for safeguard and logging

        while T > 1 and i < max_iterations:

            # Line 6: neighbor state ← SwapTwoPieces(current state)
            neighbor_state = self._get_neighbor_state(current_state)

            # Line 7: neighbor cost ← CalculateCost(neighbor state)
            neighbor_cost = self.calculate_total_cost(neighbor_state)

            # Line 8: cost diff ← neighbor cost - current cost
            delta_cost = neighbor_cost - current_cost

            # Line 9: if cost diff < 0 then
            if delta_cost < 0:
                # Line 10: current state ← neighbor state
                current_state = neighbor_state
                # Line 11: current cost ← neighbor cost
                current_cost = neighbor_cost

                # Update the overall best state found so far (not in pseudocode but essential)
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_state = current_state
                    if i % 10000 == 0:
                        print(f"Iter {i:07d}, T: {T:.2e}, New Best Cost: {best_cost:,.0f}")
            # Line 12: else
            else:
                # Line 13: acceptance prob ← e−cost diff/T
                acceptance_prob = math.exp(-delta_cost / T)

                # Line 14: if random() < acceptance prob then
                if random.random() < acceptance_prob:
                    # Line 15: current state ← neighbor state
                    current_state = neighbor_state
                    # Line 16: current cost ← neighbor cost
                    current_cost = neighbor_cost

            # Line 19: T ← T * cooling rate
            T *= cooling_rate
            i += 1

            if i % 100000 == 0 and i > 0:
                print(f"Iter {i:07d}, T: {T:.2e}, Current Cost: {current_cost:,.0f}")

        print(f"\nSA completed. Final Best Cost: {best_cost:,.0f} after {i} iterations.")
        # Line 21: return current state (returning best_state found is safer)
        return best_state

    def _assemble_full_image(self, final_state: List[int]) -> np.ndarray:
        grid_dim = self.grid_size * self.patch_dim
        reconstructed = np.zeros((grid_dim, grid_dim), dtype=np.uint8)

        # Iterate over the grid positions (r, c)
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                # Get the patch index from the 1D state array
                patch_index_in_list = final_state[r * self.grid_size + c]
                patch_to_place = self.patches[patch_index_in_list]

                # Calculate slice coordinates
                r_start, r_end = r * self.patch_dim, (r + 1) * self.patch_dim
                c_start, c_end = c * self.patch_dim, (c + 1) * self.patch_dim

                # Place the patch
                reconstructed[r_start:r_end, c_start:c_end] = patch_to_place

        return reconstructed

    def _show_image(self, image: np.ndarray, title: str):
        plt.figure(figsize=(6, 6))
        plt.imshow(image, cmap="gray", vmin=0, vmax=255)
        plt.title(title)
        plt.axis('off')
        plt.show()

    def run(self):

        # Display the completely scrambled image first
        # scrambled_image = self._assemble_full_image(list(range(self.total_tiles)))
        # self._show_image(scrambled_image, "Scrambled Image (Initial State)")

        # Run the Simulated Annealing optimization
        final_layout = self.simulated_annealing(
            max_iterations=2_500_000,
            initial_temp=100_000_000,  # Start with a very high temperature for global search
            cooling_rate=0.999995  # Slower cooling for better exploration
        )

        # Convert 1D layout back to 4x4 for display
        final_grid = np.array(final_layout).reshape((self.grid_size, self.grid_size))
        print("\nFinal Best Patch Arrangement (Indices 0-15):")
        print(final_grid)

        # Assemble and show the reconstructed image
        final_image = self._assemble_full_image(final_layout)
        self._show_image(final_image, "Reconstructed Image (Simulated Annealing Result)")


if __name__ == "__main__":
    jigsaw_solver = JigsawSolverSA("scrambled_lena.mat")
    jigsaw_solver.run()
