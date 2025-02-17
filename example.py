import logging
from qaoa_core import QAOACircuit
from qubo_formulation import QUBOFormulation
from circuit_visualization import CircuitVisualizer
from utils import Utils
import numpy as np
import random
from typing import List, Tuple, Set
from heapq import heappush, heappop

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_grid_cities(n_cities, grid_size=16):
    """Generate well-spread city coordinates on a grid."""
    positions = set()
    min_distance = grid_size // 3  # Minimum distance between cities
    quadrants = [(0, 0, grid_size//2, grid_size//2),
                (grid_size//2, 0, grid_size, grid_size//2),
                (0, grid_size//2, grid_size//2, grid_size),
                (grid_size//2, grid_size//2, grid_size, grid_size)]

    def manhattan_dist(pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def is_far_enough(new_pos):
        return all(manhattan_dist(new_pos, pos) >= min_distance for pos in positions)

    # Try to place cities in different quadrants first
    for i in range(min(n_cities, len(quadrants))):
        x_min, y_min, x_max, y_max = quadrants[i]
        attempts = 100  # Maximum attempts per quadrant
        while attempts > 0:
            x = random.randint(x_min, x_max)
            y = random.randint(y_min, y_max)
            if is_far_enough((x, y)):
                positions.add((x, y))
                break
            attempts -= 1

    # Fill remaining cities with distance constraints
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
    """Generate random larger obstacles (2x2 blocks)."""
    obstacles = set()
    total_possible_blocks = (grid_size - 1) * (grid_size - 1) // 4  # Account for 2x2 blocks
    n_obstacles = int(total_possible_blocks * obstacle_density)

    while len(obstacles) < n_obstacles * 4:  # 4 edges per obstacle
        # Generate top-left corner of 2x2 block
        x = random.randint(0, grid_size-2)
        y = random.randint(0, grid_size-2)

        # Add all edges of 2x2 block
        edges = [
            ('h', x, y),      # Top horizontal edge
            ('h', x, y+1),    # Bottom horizontal edge
            ('v', x, y),      # Left vertical edge
            ('v', x+1, y)     # Right vertical edge
        ]

        # Check if any edge already exists or if it would block a critical path
        if not any(edge in obstacles for edge in edges):
            obstacles.update(edges)

    return obstacles

def manhattan_distance_with_obstacles(city1, city2, obstacles, grid_size):
    """Calculate path distance between two cities avoiding obstacles using A* pathfinding."""
    logger = logging.getLogger(__name__)
    logger.info(f"Finding path from {city1} to {city2}")
    logger.info(f"Obstacles: {obstacles}")

    def get_neighbors(pos):
        """Get valid neighboring positions."""
        x, y = pos
        logger.debug(f"Getting neighbors for position ({x}, {y})")

        # Define moves: Up, Right, Down, Left (prioritizing vertical movement)
        moves = [
            (0, 1),   # Up
            (1, 0),   # Right
            (0, -1),  # Down
            (-1, 0)   # Left
        ]

        neighbors = []
        for dx, dy in moves:
            new_x, new_y = x + dx, y + dy

            # Check grid boundaries
            if not (0 <= new_x < grid_size and 0 <= new_y < grid_size):
                continue

            # Check obstacles based on movement direction
            blocked = False
            if dx == 0:  # Vertical movement
                if dy > 0:  # Moving up
                    blocked = ('h', x, y) in obstacles
                else:  # Moving down
                    blocked = ('h', x, y-1) in obstacles
            else:  # Horizontal movement
                if dx > 0:  # Moving right
                    blocked = ('v', x, y) in obstacles
                else:  # Moving left
                    blocked = ('v', x-1, y) in obstacles

            if not blocked:
                neighbors.append((new_x, new_y))
                logger.debug(f"Added valid move to ({new_x}, {new_y})")

        return neighbors

    def heuristic(pos1, pos2):
        """Manhattan distance heuristic with vertical preference."""
        dx = abs(pos1[0] - pos2[0])
        dy = abs(pos1[1] - pos2[1])
        return dy + dx * 1.1  # Prioritize vertical movement more strongly

    # Check if start or end is blocked
    def is_position_blocked(pos):
        x, y = pos
        return any([
            ('v', x-1, y) in obstacles,  # Left vertical wall
            ('v', x, y) in obstacles,    # Right vertical wall
            ('h', x, y-1) in obstacles,  # Bottom horizontal wall
            ('h', x, y) in obstacles     # Top horizontal wall
        ])

    if is_position_blocked(city1) or is_position_blocked(city2):
        logger.warning("Start or end position is blocked by obstacles")
        return grid_size * 10, []

    # Initialize A* search
    start = city1
    goal = city2
    frontier = [(0, start)]
    came_from = {start: None}
    cost_so_far = {start: 0}

    while frontier:
        current = min(frontier, key=lambda x: x[0])[1]
        frontier = [(f, n) for f, n in frontier if n != current]

        if current == goal:
            path = []
            pos = current
            while pos is not None:
                path.append(pos)
                pos = came_from[pos]
            path.reverse()

            # Verify path doesn't cross obstacles
            valid = True
            for i in range(len(path) - 1):
                curr_x, curr_y = path[i]
                next_x, next_y = path[i + 1]

                if curr_x == next_x:  # Vertical movement
                    if next_y > curr_y:  # Moving up
                        if ('h', curr_x, curr_y) in obstacles:
                            valid = False
                            break
                    else:  # Moving down
                        if ('h', curr_x, next_y) in obstacles:
                            valid = False
                            break
                else:  # Horizontal movement
                    if next_x > curr_x:  # Moving right
                        if ('v', curr_x, curr_y) in obstacles:
                            valid = False
                            break
                    else:  # Moving left
                        if ('v', next_x, curr_y) in obstacles:
                            valid = False
                            break

            if valid:
                logger.info(f"Found valid path: {path}")
                return len(path) - 1, path
            else:
                logger.warning("Found path crosses obstacles")
                return grid_size * 10, []

        # Check neighbors
        for next_pos in get_neighbors(current):
            new_cost = cost_so_far[current] + 1

            if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                cost_so_far[next_pos] = new_cost
                priority = new_cost + heuristic(next_pos, goal)
                frontier.append((priority, next_pos))
                came_from[next_pos] = current

        # If frontier is empty or too many positions explored, no valid path exists
        if not frontier or len(cost_so_far) >= grid_size * grid_size:
            logger.warning("No valid path found between cities")
            return grid_size * 10, []

    # No path found
    logger.warning("No valid path found between cities")
    return grid_size * 10, []

def decode_measurements(measurements, n_cities):
    """
    Decode measurements into a valid tour by selecting the maximum value
    in each row/column, with proper scaling of PauliZ measurements.
    """
    # Transform PauliZ measurements from [-1, 1] to [0, 1] range
    probabilities = [(1 + m) / 2 for m in measurements]
    measurements_matrix = np.array(probabilities).reshape((n_cities, n_cities))
    binary_solution = np.zeros_like(measurements_matrix, dtype=int)

    # First pass: Handle rows
    used_cols = set()
    for i in range(n_cities):
        # Get available columns (not used yet)
        available_cols = [j for j in range(n_cities) if j not in used_cols]
        # Select the highest probability among available columns
        probs = measurements_matrix[i, available_cols]
        if len(probs) > 0:
            j_idx = np.argmax(probs)
            j = available_cols[j_idx]
            binary_solution[i, j] = 1
            used_cols.add(j)

    # Second pass: Fix any remaining constraints
    for j in range(n_cities):
        if sum(binary_solution[:, j]) != 1:
            # Find the best unused row for this column
            used_rows = set([i for i in range(n_cities) if sum(binary_solution[i, :]) == 1])
            available_rows = [i for i in range(n_cities) if i not in used_rows]
            if available_rows:
                i = available_rows[np.argmax(measurements_matrix[available_rows, j])]
                binary_solution[:, j] = 0
                binary_solution[i, j] = 1

    return binary_solution.flatten().tolist()

def main():
    try:
        # Configuration
        n_cities = 3  # Keep at 3 cities
        grid_size = 16
        qaoa_depth = 2  # Keep depth at 2 for better convergence

        logger.info(f"Starting QAOA optimization for {n_cities} cities with depth {qaoa_depth}")

        # Generate random cities on grid
        coordinates = generate_grid_cities(n_cities, grid_size)
        logger.info("City coordinates on grid: %s", coordinates)

        # Generate obstacles with slightly lower density to account for larger obstacles
        obstacles = generate_obstacles(grid_size, obstacle_density=0.08)
        logger.info("Generated obstacles: %s", obstacles)

        # Create QUBO formulation with Manhattan distance including obstacles
        qubo = QUBOFormulation(n_cities)
        distance_matrix = np.zeros((n_cities, n_cities))
        paths_between_cities = {}  # Store paths for visualization

        for i in range(n_cities):
            for j in range(n_cities):
                distance, path = manhattan_distance_with_obstacles(
                    coordinates[i], coordinates[j], obstacles, grid_size)
                distance_matrix[i,j] = distance
                paths_between_cities[(i,j)] = path
        logger.info("\nDistance matrix:\n%s", distance_matrix)

        # Initialize circuit with more qubits
        n_qubits = n_cities * n_cities
        circuit = QAOACircuit(n_qubits, depth=qaoa_depth)

        # Create cost terms from QUBO matrix
        qubo_matrix = qubo.create_qubo_matrix(distance_matrix, penalty=10.0)
        cost_terms = []
        for i in range(n_qubits):
            for j in range(i, n_qubits):
                if abs(qubo_matrix[i, j]) > 1e-10:
                    cost_terms.append((float(qubo_matrix[i, j]), (i, j)))

        logger.info("Number of cost terms: %d", len(cost_terms))

        # Run optimization with timeout
        logger.info("Starting QAOA optimization...")
        params, cost_history = circuit.optimize(cost_terms, steps=100, timeout=120)  # 2 minutes timeout
        logger.info("Optimization completed")

        # Get measurements and decode solution
        measurements = circuit.circuit(params, cost_terms)
        binary_solution = decode_measurements(measurements, n_cities)

        route = qubo.decode_solution(binary_solution)

        if Utils.verify_solution(route, n_cities):
            # Calculate total route length considering obstacles
            route_length = 0
            route_paths = []  # Collect all paths for visualization
            for i in range(n_cities):
                start = route[i]
                end = route[(i+1)%n_cities]
                path = paths_between_cities[(start, end)]
                route_paths.append(path)
                route_length += distance_matrix[start, end]

            logger.info("\nValid solution found!")
            logger.info("Route: %s", ' -> '.join(str(x) for x in route))
            logger.info("Route length: %d", int(route_length))

            # Save visualizations to files
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