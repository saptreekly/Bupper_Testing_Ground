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

def generate_obstacles(grid_size=16, obstacle_density=0.1):
    obstacles = set()
    total_possible_blocks = (grid_size - 1) * (grid_size - 1) // 4
    n_obstacles = int(total_possible_blocks * obstacle_density)

    while len(obstacles) < n_obstacles * 4:
        x = random.randint(0, grid_size-2)
        y = random.randint(0, grid_size-2)

        edges = [
            ('h', x, y),
            ('h', x, y+1),
            ('v', x, y),
            ('v', x+1, y)
        ]

        if not any(edge in obstacles for edge in edges):
            obstacles.update(edges)

    return obstacles

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
        parser = argparse.ArgumentParser(description='QAOA Routing Optimizer')
        parser.add_argument('--coordinates', type=str, help='City coordinates in format "x1,y1;x2,y2;x3,y3"')
        parser.add_argument('--cities', type=int, default=3, help='Number of cities (default: 3)')
        parser.add_argument('--grid-size', type=int, default=16, help='Grid size (default: 16)')
        parser.add_argument('--qaoa-depth', type=int, default=2, help='QAOA circuit depth (default: 2)')
        args = parser.parse_args()

        n_cities = args.cities
        grid_size = args.grid_size
        qaoa_depth = args.qaoa_depth

        logger.info(f"Starting QAOA optimization for {n_cities} cities with depth {qaoa_depth}")

        coordinates = None

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
            try:
                custom_coords = input("Enter city coordinates (format: x1,y1;x2,y2;x3,y3) or press Enter for random: ")
                if custom_coords.strip():
                    coordinates = parse_coordinates(custom_coords)
                    if not coordinates or len(coordinates) != n_cities:
                        logger.error(f"Invalid number of coordinates. Expected {n_cities}, got {len(coordinates)}")
                        return
                    if not validate_coordinates(coordinates, grid_size):
                        logger.error("Invalid coordinates provided")
                        return
                    logger.info("Using custom coordinates: %s", coordinates)
            except (EOFError, KeyboardInterrupt):
                logger.info("No input provided or running in non-interactive mode")
                coordinates = None

        if coordinates is None:
            coordinates = generate_grid_cities(n_cities, grid_size)
            logger.info("Using generated coordinates: %s", coordinates)

        obstacles = generate_obstacles(grid_size, obstacle_density=0.08)
        logger.info("Generated obstacles: %s", obstacles)

        qubo = QUBOFormulation(n_cities)
        distance_matrix = np.zeros((n_cities, n_cities))
        paths_between_cities = {}

        for i in range(n_cities):
            for j in range(n_cities):
                distance, path = manhattan_distance_with_obstacles(
                    coordinates[i], coordinates[j], obstacles, grid_size)
                distance_matrix[i,j] = distance
                paths_between_cities[(i,j)] = path
        logger.info("\nDistance matrix:\n%s", distance_matrix)

        n_qubits = n_cities * n_cities
        circuit = QAOACircuit(n_qubits, depth=qaoa_depth)

        qubo_matrix = qubo.create_qubo_matrix(distance_matrix, penalty=10.0)
        cost_terms = []
        for i in range(n_qubits):
            for j in range(i, n_qubits):
                if abs(qubo_matrix[i, j]) > 1e-10:
                    cost_terms.append((float(qubo_matrix[i, j]), (i, j)))

        logger.info("Number of cost terms: %d", len(cost_terms))

        logger.info("Starting QAOA optimization...")
        params, cost_history = circuit.optimize(cost_terms, steps=100, timeout=120)
        logger.info("Optimization completed")

        measurements = circuit.circuit(params, cost_terms)
        binary_solution = decode_measurements(measurements, n_cities)

        route = qubo.decode_solution(binary_solution)

        if Utils.verify_solution(route, n_cities):
            route_length = 0
            route_paths = []
            for i in range(n_cities):
                start = route[i]
                end = route[(i+1)%n_cities]
                path = paths_between_cities[(start, end)]
                route_paths.append(path)
                route_length += distance_matrix[start, end]

            logger.info("\nValid solution found!")
            logger.info("Route: %s", ' -> '.join(str(x) for x in route))
            logger.info("Route length: %d", int(route_length))

            visualizer = CircuitVisualizer()
            visualizer.plot_route(coordinates, route, grid_size=grid_size, 
                                   obstacles=obstacles, route_paths=route_paths,
                                   save_path="route.png")
            visualizer.plot_optimization_trajectory(cost_history, 
                                                     save_path="optimization_trajectory.png")
            logger.info("Visualizations saved as 'route.png' and 'optimization_trajectory.png'")
        else:
            logger.warning("Invalid solution found. Binary solution: %s", binary_solution)

    except Exception as e:
        logger.error("Error in main: %s", str(e), exc_info=True)
        raise

if __name__ == "__main__":
    main()