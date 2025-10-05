import random
import math
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Callable

DESTINATION_COORDINATES: Dict[str, Tuple[float, float]] = {
    "Amber Fort": (26.9855, 75.8513),
    "City Palace": (24.5764, 73.6844),
    "Mehrangarh Fort": (26.2983, 73.0193),
    "Brahma Temple": (26.4872, 74.5503),
    "Golden Fort": (26.9115, 70.9177),
    "Dargah Sharif": (26.4519, 74.6275),
    "Dilwara Temples": (24.6001, 72.7145),
    "Junagarh Fort": (28.0121, 73.3187),
    "Ranthambore National Park": (26.0173, 76.5026),
    "Chittorgarh Fort": (24.8879, 74.6454),
    "Taragarh Fort": (25.4359, 75.6473),
    "Bhangarh Fort": (27.0964, 76.2850),
    "Keoladeo National Park": (27.1672, 77.5222),
    "Seven Wonders Park": (25.1602, 75.8510),
    "Ranthambore Fort": (26.0208, 76.4569),
    "Nawalgarh": (27.8513, 75.2739),
    "Juna Mahal": (23.8359, 73.7148),
    "Shrinathji Temple": (24.9268, 73.8315),
    "Mandawa Castle": (28.0559, 75.1545),
    "Sachiya Mata Temple": (26.9128, 72.3941)
}

# Earth radius constant for distance calculation
EARTH_RADIUS_KM = 6371

def haversine_distance(coords_a: Tuple[float, float], coords_b: Tuple[float, float]) -> float:
    lat1, lon1 = coords_a
    lat2, lon2 = coords_b
    
    # Convert degrees to radians
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    # Haversine formula
    sin_sq_dphi = math.sin(delta_phi / 2)**2
    sin_sq_dlambda = math.sin(delta_lambda / 2)**2
    
    a = sin_sq_dphi + math.cos(phi1) * math.cos(phi2) * sin_sq_dlambda
    
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return EARTH_RADIUS_KM * c

def calculate_tour_length(path: List[str], coordinates_map: Dict[str, Tuple[float, float]]) -> float:
    total_length = 0.0
    num_sites = len(path)
    
    for i in range(num_sites):
        site_a = path[i]
        # The modulo operator ensures it connects the last site back to the first
        site_b = path[(i + 1) % num_sites] 
        
        coords_a = coordinates_map[site_a]
        coords_b = coordinates_map[site_b]
        
        total_length += haversine_distance(coords_a, coords_b)
        
    return total_length


def annealing_tsp_solver(
    site_coords: Dict[str, Tuple[float, float]], 
    initial_T: float = 2000.0, 
    cooling_factor: float = 0.995, 
    max_steps: int = 150000
) -> Tuple[List[str], float, List[float]]:
    sites_list = list(site_coords.keys())
    
    # Start with a random permutation
    current_path = sites_list.copy()
    random.shuffle(current_path)
    
    current_cost = calculate_tour_length(current_path, site_coords)
    best_path = current_path.copy()
    best_cost = current_cost
    
    T = initial_T
    cost_log = []

    for step in range(max_steps):
        # 1. Generate a neighbor state (Two-opt swap is a common, more effective neighbor, but sticking to the original simple swap for output consistency)
        new_path = current_path.copy()
        idx1, idx2 = random.sample(range(len(new_path)), 2)
        new_path[idx1], new_path[idx2] = new_path[idx2], new_path[idx1] # Simple 2-swap
        
        new_cost = calculate_tour_length(new_path, site_coords)
        
        cost_delta = new_cost - current_cost
        
        # 2. Acceptance criterion
        if cost_delta < 0 or random.random() < math.exp(-cost_delta / T):
            current_path = new_path
            current_cost = new_cost
            
            # 3. Update global best
            if new_cost < best_cost:
                best_path = new_path.copy()
                best_cost = new_cost

        # 4. Log and Cool
        cost_log.append(current_cost)
        T *= cooling_factor
        
    return best_path, best_cost, cost_log


def plot_tour_map(tour_path: List[str], coordinates_map: Dict[str, Tuple[float, float]]):
    plt.figure(figsize=(12, 10))
    
    # Extract coordinates for plotting and closing the loop
    tour_lons = [coordinates_map[site][1] for site in tour_path] + [coordinates_map[tour_path[0]][1]]
    tour_lats = [coordinates_map[site][0] for site in tour_path] + [coordinates_map[tour_path[0]][0]]
    
    # Plot connections
    plt.plot(tour_lons, tour_lats, 'r-', linewidth=1.5, zorder=1)
    
    # Plot sites and add labels
    for site, (lat, lon) in coordinates_map.items():
        plt.plot(lon, lat, 'ko', markersize=6, zorder=2) # Black circles for clarity
        plt.text(lon, lat, site, fontsize=8, ha='right', va='bottom', color='darkblue')

    plt.title('Optimized Rajasthan Tourist Circuit (Simulated Annealing)')
    plt.xlabel('Longitude ($\lambda$)')
    plt.ylabel('Latitude ($\phi$)')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()

def plot_optimization_progress(history: List[float]):
    plt.figure(figsize=(10, 6))
    plt.plot(history, color='purple', alpha=0.8)
    plt.title('Optimization Progress: Tour Distance vs. Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Current Tour Distance (km)')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()


if __name__ == "__main__":
    
    # Run the optimization
    best_tour, min_distance, distance_history = annealing_tsp_solver(DESTINATION_COORDINATES)

    # Calculate individual segment distances for detailed output
    segment_distances = []
    num_segments = len(best_tour)
    for i in range(num_segments):
        start_site = best_tour[i]
        end_site = best_tour[(i + 1) % num_segments]
        coords_start = DESTINATION_COORDINATES[start_site]
        coords_end = DESTINATION_COORDINATES[end_site]
        dist = haversine_distance(coords_start, coords_end)
        segment_distances.append((start_site, end_site, dist))

    print("Optimized Rajasthan Tour (Simulated Annealing)")
    
    print("\nOptimal Tour Sequence:")
    for i, site in enumerate(best_tour):
        print(f" {i+1:02d}. {site}")
        
    print(f"\nTotal Minimum Tour Distance: {min_distance:.2f} km")
    
    print("\nSegment Distances:")
    for start, end, dist in segment_distances:
        print(f" {start} to {end}: {dist:.2f} km")
        
    plot_tour_map(best_tour, DESTINATION_COORDINATES)
    plot_optimization_progress(distance_history)