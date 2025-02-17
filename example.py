import logging
from qaoa_core import QAOACircuit
from qubo_formulation import QUBOFormulation
from circuit_visualization import CircuitVisualizer
from utils import Utils
import numpy as np
import random

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_grid_cities(n_cities, grid_size=16):
    """Generate random city coordinates on a grid."""
    positions = set()
    while len(positions) < n_cities:
        x = np.random.randint(0, grid_size)
        y = np.random.randint(0, grid_size)
        positions.add((x, y))
    return list(positions)

def generate_obstacles(grid_size=16, obstacle_density=0.1):
    """Generate random obstacles (blocked grid lines)."""
    obstacles = set()
    total_possible_lines = (grid_size + 1) * grid_size * 2  # Vertical and horizontal lines
    n_obstacles = int(total_possible_lines * obstacle_density)

    while len(obstacles) < n_obstacles:
        # Randomly choose horizontal or vertical line
        is_horizontal = random.random() < 0.5
        if is_horizontal:
            x = random.randint(0, grid_size-1)
            y = random.randint(0, grid_size)
            obstacles.add(('h', x, y))  # Horizontal line at (x,y) to (x+1,y)
        else:
            x = random.randint(0, grid_size)
            y = random.randint(0, grid_size-1)
            obstacles.add(('v', x, y))  # Vertical line at (x,y) to (x,y+1)

    return obstacles

def manhattan_distance_with_obstacles(city1, city2, obstacles, grid_size):
    """Calculate Manhattan distance between two cities avoiding obstacles."""
    def is_path_blocked(x1, y1, x2, y2):
        # Check if the path between two adjacent points is blocked
        if x1 == x2:  # Vertical movement
            min_y, max_y = min(y1, y2), max(y1, y2)
            return ('v', x1, min_y) in obstacles
        else:  # Horizontal movement
            min_x, max_x = min(x1, x2), max(x1, x2)
            return ('h', min_x, y1) in obstacles

    # Simple implementation: if direct Manhattan path is blocked,
    # use a penalty factor to encourage finding alternative routes
    x1, y1 = city1
    x2, y2 = city2
    direct_distance = abs(x2 - x1) + abs(y2 - y1)

    # Check for blocked paths in the direct route
    blocked_count = 0
    # Check horizontal movement
    for x in range(min(x1, x2), max(x1, x2)):
        if is_path_blocked(x, y1, x+1, y1):
            blocked_count += 1
    # Check vertical movement
    for y in range(min(y1, y2), max(y1, y2)):
        if is_path_blocked(x2, y, x2, y+1):
            blocked_count += 1

    # Add penalty for blocked paths
    return direct_distance + blocked_count * grid_size

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
        n_cities = 5  # Increased number of cities
        grid_size = 16
        qaoa_depth = 2  # Reduced depth for faster convergence

        logger.info(f"Starting QAOA optimization for {n_cities} cities with depth {qaoa_depth}")

        # Generate random cities on grid
        coordinates = generate_grid_cities(n_cities, grid_size)
        logger.info("City coordinates on grid: %s", coordinates)

        # Generate obstacles
        obstacles = generate_obstacles(grid_size, obstacle_density=0.1)
        logger.info("Generated %d obstacles", len(obstacles))

        # Create QUBO formulation with Manhattan distance including obstacles
        qubo = QUBOFormulation(n_cities)
        distance_matrix = np.zeros((n_cities, n_cities))
        for i in range(n_cities):
            for j in range(n_cities):
                distance_matrix[i,j] = manhattan_distance_with_obstacles(
                    coordinates[i], coordinates[j], obstacles, grid_size)
        logger.info("\nDistance matrix:\n%s", distance_matrix)

        # Increased penalty for stronger constraint satisfaction
        qubo_matrix = qubo.create_qubo_matrix(distance_matrix, penalty=10.0)

        # Initialize circuit with more qubits
        n_qubits = n_cities * n_cities
        circuit = QAOACircuit(n_qubits, depth=qaoa_depth)

        # Create cost terms from QUBO matrix
        cost_terms = []
        for i in range(n_qubits):
            for j in range(i, n_qubits):
                if abs(qubo_matrix[i, j]) > 1e-10:
                    cost_terms.append((float(qubo_matrix[i, j]), (i, j)))

        logger.info("Number of cost terms: %d", len(cost_terms))

        # Run optimization with timeout
        logger.info("Starting QAOA optimization...")
        params, cost_history = circuit.optimize(cost_terms, steps=150, timeout=180)  # 3 minutes timeout
        logger.info("Optimization completed")

        # Get measurements and decode solution
        measurements = circuit.circuit(params, cost_terms)
        binary_solution = decode_measurements(measurements, n_cities)

        route = qubo.decode_solution(binary_solution)

        if Utils.verify_solution(route, n_cities):
            # Calculate total route length considering obstacles
            route_length = 0
            for i in range(n_cities):
                start = coordinates[route[i]]
                end = coordinates[route[(i+1)%n_cities]]
                route_length += manhattan_distance_with_obstacles(
                    start, end, obstacles, grid_size)

            logger.info("\nValid solution found!")
            logger.info("Route: %s", ' -> '.join(str(x) for x in route))
            logger.info("Route length: %d", route_length)

            # Save visualizations to files
            visualizer = CircuitVisualizer()
            visualizer.plot_route(coordinates, route, grid_size=grid_size, 
                                   obstacles=obstacles, save_path="route.png")
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