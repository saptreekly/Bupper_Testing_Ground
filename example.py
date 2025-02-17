import logging
from qaoa_core import QAOACircuit
from qubo_formulation import QUBOFormulation
from circuit_visualization import CircuitVisualizer
from utils import Utils
import numpy as np
import random
from typing import List, Tuple, Set
from heapq import heappush, heappop
import argparse
import sys
import matplotlib
from classical_solver import solve_tsp_brute_force
from classical_solver import clarke_wright_savings
import time

matplotlib.use('Agg')  # Force non-interactive backend

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_grid_cities(n_cities, grid_size=16):
    positions = set()
    min_distance = grid_size // 3
    quadrants = [(0, 0, grid_size//2, grid_size//2),
                (grid_size//2, 0, grid_size, grid_size//2),
                (0, grid_size//2, grid_size//2, grid_size),
                (grid_size//2, grid_size//2, grid_size, grid_size)]

    def manhattan_dist(pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def is_far_enough(new_pos):
        return all(manhattan_dist(new_pos, pos) >= min_distance for pos in positions)

    for i in range(min(n_cities, len(quadrants))):
        x_min, y_min, x_max, y_max = quadrants[i]
        attempts = 100
        while attempts > 0:
            x = random.randint(x_min, x_max)
            y = random.randint(y_min, y_max)
            if is_far_enough((x, y)):
                positions.add((x, y))
                break
            attempts -= 1

    while len(positions) < n_cities:
        x = random.randint(0, grid_size)
        y = random.randint(0, grid_size)
        if is_far_enough((x, y)):
            positions.add((x, y))

    positions_list = list(positions)
    logger.info("Generated city coordinates: %s", positions_list)
    for i in range(len(positions_list)):
        for j in range(i+1, len(positions_list)):
            dist = manhattan_dist(positions_list[i], positions_list[j])
            logger.info("Distance between cities %d and %d: %d units", i, j, dist)

    return positions_list

def generate_random_demands(n_cities: int, min_demand: float = 1.0, 
                          max_demand: float = 5.0, depot_index: int = 0) -> List[float]:
    """Generate random demands for each city (depot demand is 0)."""
    demands = [random.uniform(min_demand, max_demand) for _ in range(n_cities)]
    demands[depot_index] = 0.0  # Depot has no demand
    return demands

def manhattan_distance_with_obstacles(city1, city2, obstacles, grid_size):
    logger.info(f"Finding path from {city1} to {city2}")

    path = []
    current = list(city1)
    target = list(city2)

    while current[0] != target[0]:
        if current[0] < target[0]:
            current[0] += 1
        else:
            current[0] -= 1
        path.append(tuple(current))

    while current[1] != target[1]:
        if current[1] < target[1]:
            current[1] += 1
        else:
            current[1] -= 1
        path.append(tuple(current))

    path.insert(0, city1)

    logger.info(f"Generated path: {path}")
    return len(path) - 1, path

def decode_measurements(measurements, n_cities):
    probabilities = [(1 + m) / 2 for m in measurements]
    measurements_matrix = np.array(probabilities).reshape((n_cities, n_cities))
    binary_solution = np.zeros_like(measurements_matrix, dtype=int)

    used_cols = set()
    for i in range(n_cities):
        available_cols = [j for j in range(n_cities) if j not in used_cols]
        probs = measurements_matrix[i, available_cols]
        if len(probs) > 0:
            j_idx = np.argmax(probs)
            j = available_cols[j_idx]
            binary_solution[i, j] = 1
            used_cols.add(j)

    for j in range(n_cities):
        if sum(binary_solution[:, j]) != 1:
            used_rows = set([i for i in range(n_cities) if sum(binary_solution[i, :]) == 1])
            available_rows = [i for i in range(n_cities) if i not in used_rows]
            if available_rows:
                i = available_rows[np.argmax(measurements_matrix[available_rows, j])]
                binary_solution[:, j] = 0
                binary_solution[i, j] = 1

    return binary_solution.flatten().tolist()

def validate_coordinates(coordinates: List[Tuple[int, int]], grid_size: int) -> bool:
    seen = set()
    for x, y in coordinates:
        if not (0 <= x < grid_size and 0 <= y < grid_size):
            logger.warning(f"Coordinates ({x}, {y}) out of bounds [0, {grid_size})")
            return False
        if (x, y) in seen:
            logger.warning(f"Duplicate coordinates found: ({x}, {y})")
            return False
        seen.add((x, y))
    return True

def parse_coordinates(coord_str: str) -> List[Tuple[int, int]]:
    try:
        pairs = coord_str.strip().split(';')
        coordinates = []
        for pair in pairs:
            x_str, y_str = pair.strip().split(',')
            x, y = int(x_str), int(y_str)
            coordinates.append((x, y))
        return coordinates
    except Exception as e:
        logger.error(f"Failed to parse coordinates: {str(e)}")
        return []

def main():
    try:
        parser = argparse.ArgumentParser(description='QAOA Vehicle Routing Optimizer')
        parser.add_argument('--coordinates', type=str, help='City coordinates in format "x1,y1;x2,y2;x3,y3"')
        parser.add_argument('--cities', type=int, default=4, help='Number of cities (default: 4)')  # Changed to 4 cities
        parser.add_argument('--grid-size', type=int, default=16, help='Grid size (default: 16)')
        parser.add_argument('--qaoa-depth', type=int, default=1, help='QAOA circuit depth (default: 1)')
        parser.add_argument('--vehicles', type=int, default=1, help='Number of vehicles (default: 1)')
        parser.add_argument('--capacity', type=float, default=10.0, help='Vehicle capacity (default: 10.0)')
        parser.add_argument('--non-interactive', action='store_true', help='Run in non-interactive mode')
        args = parser.parse_args()

        n_cities = args.cities
        grid_size = args.grid_size
        qaoa_depth = args.qaoa_depth
        n_vehicles = args.vehicles
        vehicle_capacity = [args.capacity] * n_vehicles

        # Check if problem size is too large
        n_qubits = n_cities * n_cities * n_vehicles
        if n_qubits > 16:  # Adjusted for 4 cities (16 qubits)
            logger.error(f"Problem size too large: {n_qubits} qubits required. Please reduce number of cities or vehicles.")
            return

        logger.info(f"Starting QAOA optimization for {n_cities} cities with {n_vehicles} vehicles")
        logger.info("Note: Larger problem sizes may require more optimization steps")

        # Handle coordinates input
        if args.coordinates:
            coordinates = parse_coordinates(args.coordinates)
            if not coordinates or len(coordinates) != n_cities:
                logger.error(f"Invalid number of coordinates. Expected {n_cities}, got {len(coordinates)}")
                return
            if not validate_coordinates(coordinates, grid_size):
                logger.error("Invalid coordinates provided")
                return
            logger.info("Using command line coordinates: %s", coordinates)
        else:
            coordinates = generate_grid_cities(n_cities, grid_size)
            logger.info("Using generated coordinates: %s", coordinates)

        # Generate random demands for cities
        demands = generate_random_demands(n_cities)
        logger.info("City demands: %s", demands)

        qubo = QUBOFormulation(n_cities, n_vehicles, vehicle_capacity)
        distance_matrix = np.zeros((n_cities, n_cities))
        paths_between_cities = {}

        for i in range(n_cities):
            for j in range(n_cities):
                if i != j:
                    distance, path = manhattan_distance_with_obstacles(coordinates[i], coordinates[j], set(), grid_size)
                    distance_matrix[i,j] = distance
                    paths_between_cities[(i,j)] = path
        logger.info("\nDistance matrix:\n%s", distance_matrix)

        # Compare with classical solutions
        logger.info("\nSolving with classical methods...")

        # Time the classical solvers
        start_time = time.time()
        optimal_route, optimal_length = solve_tsp_brute_force(distance_matrix)
        brute_force_time = time.time() - start_time

        start_time = time.time()
        cw_routes, cw_length = clarke_wright_savings(distance_matrix, demands, 
                                                   depot_index=0, capacity=vehicle_capacity[0])
        cw_time = time.time() - start_time

        # Time the quantum solution
        start_time = time.time()
        # Initialize circuit with minimal parameters
        circuit = QAOACircuit(n_qubits, depth=1)
        logger.info("Initialized QAOA circuit with %d qubits", n_qubits)

        # Create cost terms with increased threshold
        qubo_matrix = qubo.create_qubo_matrix(distance_matrix, demands=demands, penalty=0.5)  # Reduced penalty
        cost_terms = []
        for i in range(n_qubits):
            for j in range(i, n_qubits):
                if abs(qubo_matrix[i, j]) > 1e-2:  # Increased threshold
                    cost_terms.append((float(qubo_matrix[i, j]), (i, j)))

        logger.info("Number of cost terms: %d", len(cost_terms))
        logger.info("Starting QAOA optimization...")

        # Run optimization with minimal parameters
        params, costs = circuit.optimize(cost_terms, steps=10)
        measurements = circuit.circuit(params, cost_terms)
        binary_solution = decode_measurements(measurements, n_cities)
        routes = qubo.decode_solution(binary_solution)
        quantum_time = time.time() - start_time

        # Calculate total quantum route length
        total_route_length = 0
        all_route_paths = []
        for vehicle_idx, route in enumerate(routes):
            route_length = 0
            route_paths = []
            route_demand = 0
            for i in range(len(route)-1):
                start = route[i]
                end = route[i+1]
                path = paths_between_cities[(start, end)]
                route_paths.append(path)
                route_length += distance_matrix[start, end]
                if i > 0:
                    route_demand += demands[start]
            total_route_length += route_length
            all_route_paths.extend(route_paths)

        # Prepare metrics for visualization
        quantum_metrics = {
            "distance": total_route_length,
            "time": quantum_time
        }

        classical_metrics = {
            "distance": cw_length,
            "time": cw_time
        }

        # Compare solutions
        logger.info("\nSolution comparison:")
        logger.info(f"Quantum solution length: {total_route_length:.2f} (time: {quantum_time:.2f}s)")
        logger.info(f"Clarke-Wright solution length: {cw_length:.2f} (time: {cw_time:.2f}s)")
        logger.info(f"Brute force optimal length: {optimal_length:.2f} (time: {brute_force_time:.2f}s)")

        # Visualization
        visualizer = CircuitVisualizer()
        try:
            # Original route visualization
            visualizer.plot_route(coordinates, routes, grid_size=grid_size, 
                                obstacles=set(), route_paths=all_route_paths,
                                save_path="quantum_route.png")

            # Optimization trajectory
            visualizer.plot_optimization_trajectory(costs, 
                                                save_path="optimization_trajectory.png")

            # Compare quantum vs classical
            visualizer.plot_solution_comparison(coordinates, 
                                             routes[0], cw_routes[0],  # Compare first route
                                             quantum_metrics, classical_metrics,
                                             save_path="solution_comparison.png")

            logger.info("Visualizations saved as 'quantum_route.png', 'optimization_trajectory.png', "
                       "and 'solution_comparison.png'")

        except Exception as viz_error:
            logger.error("Visualization error: %s", str(viz_error))

    except Exception as e:
        logger.error("Error in main: %s", str(e), exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()