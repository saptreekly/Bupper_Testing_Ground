import logging
from qaoa_core import QAOACircuit
from qiskit_qaoa import QiskitQAOA
from qubo_formulation import QUBOFormulation
from circuit_visualization import CircuitVisualizer
from street_network import StreetNetwork
from utils import Utils
import numpy as np
import random
from typing import List, Tuple, Set, Dict, Any
from heapq import heappush, heappop
import argparse
import sys
import matplotlib
from classical_solver import solve_tsp_brute_force
from classical_solver import clarke_wright_savings
import time
from hybrid_optimizer import HybridOptimizer
import osmnx as ox

matplotlib.use('Agg')

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
    demands = [random.uniform(min_demand, max_demand) for _ in range(n_cities)]
    demands[depot_index] = 0.0
    return demands

def manhattan_distance_with_obstacles(city1, city2, obstacles, grid_size, network):
    """Calculate path distance between two points using lat/long coordinates."""
    logger.info(f"Finding path from {city1} to {city2}")

    try:
        # Calculate direct distance using haversine formula
        direct_distance = network.calculate_haversine_distance(
            city1[0], city1[1],  # lat1, lon1
            city2[0], city2[1]   # lat2, lon2
        )

        # Generate intermediate points for visualization
        num_points = 10
        path = []

        # Linear interpolation between points
        for i in range(num_points + 1):
            t = i / num_points
            lat = city1[0] + t * (city2[0] - city1[0])
            lon = city1[1] + t * (city2[1] - city1[1])
            path.append((lat, lon))

        logger.info(f"Generated path with {len(path)} points and total distance {direct_distance:.1f}m")
        return direct_distance, path

    except Exception as e:
        logger.error(f"Error calculating path distance: {str(e)}")
        # Return a very large distance in case of error
        return float('inf'), [city1, city2]

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

import time
from typing import Dict, Any
import numpy as np

def benchmark_optimization(n_cities: int, n_vehicles: int, place_name: str,
                         backend: str, hybrid: bool = False, check_cancelled=None) -> Dict[str, Any]:
    metrics = {}
    start_time = time.time()

    try:
        coordinates, nodes, network = generate_street_network_cities(n_cities, place_name)
        demands = generate_random_demands(n_cities)
        qubo = QUBOFormulation(n_cities, n_vehicles, [float('inf')] * n_vehicles)
        distance_matrix = network.get_distance_matrix(nodes)

        # Check for cancellation after network setup
        if check_cancelled and check_cancelled():
            raise RuntimeError("Optimization cancelled by user")

        metrics['problem_setup_time'] = time.time() - start_time
        metrics['problem_size'] = {
            'n_cities': n_cities,
            'n_vehicles': n_vehicles,
            'n_qubits': n_cities * n_cities * n_vehicles
        }
        logger.info(f"Problem size metrics: {metrics['problem_size']}")

        start_time = time.time()
        qubo_matrix = qubo.create_qubo_matrix(distance_matrix, demands=demands, penalty=2.0)
        metrics['qubo_formation_time'] = time.time() - start_time
        metrics['qubo_sparsity'] = np.count_nonzero(qubo_matrix) / (qubo_matrix.size)
        logger.info(f"QUBO formation metrics - Time: {metrics['qubo_formation_time']:.3f}s, Sparsity: {metrics['qubo_sparsity']:.3f}")

        # Check for cancellation after QUBO formation
        if check_cancelled and check_cancelled():
            raise RuntimeError("Optimization cancelled by user")

        start_time = time.time()
        n_qubits = n_cities * n_cities * n_vehicles
        cost_terms = []
        max_coeff = np.max(np.abs(qubo_matrix))
        mean_coeff = np.mean(np.abs(qubo_matrix[np.nonzero(qubo_matrix)]))
        threshold = mean_coeff * 0.01

        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                if abs(qubo_matrix[i, j]) > threshold:
                    cost_terms.append((float(qubo_matrix[i, j]), (i, j)))

            # Check for cancellation periodically during cost terms generation
            if i % 100 == 0 and check_cancelled and check_cancelled():
                raise RuntimeError("Optimization cancelled by user")

        metrics['cost_terms_generation_time'] = time.time() - start_time
        metrics['n_cost_terms'] = len(cost_terms)
        metrics['cost_terms_density'] = len(cost_terms) / (n_qubits * (n_qubits - 1) / 2)
        logger.info(f"Generated {len(cost_terms)} cost terms with density {metrics['cost_terms_density']:.3f}")

        circuit_start = time.time()
        if hybrid:
            circuit = HybridOptimizer(n_qubits, depth=min(2, n_cities//2), backend=backend)
            logger.info(f"Initialized hybrid optimizer with {backend} backend")
        else:
            if backend == 'qiskit':
                circuit = QiskitQAOA(n_qubits, depth=min(2, n_cities//2))
                logger.info(f"Initialized qiskit circuit with {n_qubits} qubits")
            else:
                circuit = QAOACircuit(n_qubits, depth=min(2, n_cities//2))
                logger.info(f"Initialized {backend} circuit with {n_qubits} qubits")

        metrics['circuit_initialization_time'] = time.time() - circuit_start

        # Check for cancellation before optimization
        if check_cancelled and check_cancelled():
            raise RuntimeError("Optimization cancelled by user")

        optimization_start = time.time()
        steps = min(100, n_qubits * 5)

        # Backend-specific optimization
        if backend == 'qiskit':
            # For Qiskit backend, we use its native interface
            params, costs = circuit.optimize(cost_terms, steps=steps)
            # Check cancellation after optimization
            if check_cancelled and check_cancelled():
                raise RuntimeError("Optimization cancelled by user")
        else:
            # For PennyLane backend, we use direct optimization without callback
            params, costs = circuit.optimize(cost_terms, steps=steps)
            # Check cancellation after optimization is complete
            if check_cancelled and check_cancelled():
                raise RuntimeError("Optimization cancelled by user")

        metrics['optimization_time'] = time.time() - optimization_start
        logger.info(f"Optimization completed in {metrics['optimization_time']:.3f}s")

        solution_start = time.time()
        measurements = circuit.get_expectation_values(params, cost_terms)
        binary_solution = [1 if x > 0 else 0 for x in measurements]
        routes = qubo.decode_solution(binary_solution)

        # Store routes in metrics
        metrics['routes'] = routes
        metrics['network'] = network
        metrics['nodes'] = nodes

        total_length = 0
        max_route_length = 0
        for route in routes:
            route_length = 0
            for i in range(len(route)-1):
                route_length += distance_matrix[route[i], route[i+1]]
            total_length += route_length
            max_route_length = max(max_route_length, route_length)

        metrics['solution_computation_time'] = time.time() - solution_start
        metrics['total_time'] = time.time() - start_time
        metrics['solution_length'] = total_length
        metrics['max_route_length'] = max_route_length
        metrics['n_routes'] = len(routes)
        metrics['convergence_history'] = costs
        metrics['final_cost'] = costs[-1]

        classical_start = time.time()
        _, classical_length = clarke_wright_savings(distance_matrix, demands, 
                                                depot_index=0, capacity=float('inf'))
        metrics['classical_solution_time'] = time.time() - classical_start
        metrics['quantum_classical_gap'] = (total_length - classical_length) / classical_length

        logger.info(f"Solution metrics - Length: {total_length:.2f}, Gap to classical: {metrics['quantum_classical_gap']:.1%}")
        return metrics

    except Exception as e:
        logger.error(f"Error in benchmark optimization: {str(e)}", exc_info=True)
        raise

def generate_street_network_cities(n_cities: int, place_name: str = "San Francisco, California, USA") -> Tuple[List[Tuple[float, float]], List[int], StreetNetwork]:
    network = StreetNetwork(place_name)
    selected_nodes = network.get_random_nodes(n_cities)
    coordinates = network.get_node_coordinates(selected_nodes)

    return coordinates, selected_nodes, network

def main():
    try:
        parser = argparse.ArgumentParser(description='QAOA Vehicle Routing Optimizer')
        parser.add_argument('--coordinates', type=str, help='City coordinates in format "x1,y1;x2,y2;x3,y3"')
        parser.add_argument('--cities', type=int, default=4, help='Number of cities (default: 4)')
        parser.add_argument('--grid-size', type=int, default=16, help='Grid size (default: 16)')
        parser.add_argument('--qaoa-depth', type=int, default=1, help='QAOA circuit depth (default: 1)')
        parser.add_argument('--vehicles', type=int, default=1, help='Number of vehicles (default: 1)')
        parser.add_argument('--capacity', type=float, default=10.0, help='Vehicle capacity (default: 10.0)')
        parser.add_argument('--non-interactive', action='store_true', help='Run in non-interactive mode')
        parser.add_argument('--backend', choices=['pennylane', 'qiskit'], default='qiskit',
                          help='Choose quantum backend (default: qiskit)')
        parser.add_argument('--hybrid', action='store_true',
                          help='Use hybrid quantum-classical optimization')
        parser.add_argument('--benchmark', action='store_true',
                          help='Run benchmarking suite')
        parser.add_argument('--location', type=str, default="San Francisco, California, USA",
                          help='Location name for street network (default: San Francisco)')
        args = parser.parse_args()

        if args.benchmark:
            logger.info("\nRunning benchmarking suite...")
            problem_sizes = [(3, 1), (4, 1), (5, 1)]
            backends = ['qiskit', 'pennylane']
            all_metrics = []

            visualizer = CircuitVisualizer()

            for n_cities, n_vehicles in problem_sizes:
                for backend in backends:
                    for hybrid in [False, True]:
                        logger.info(f"\nBenchmarking {n_cities} cities, {n_vehicles} vehicles "
                                  f"with {backend} backend, hybrid={hybrid}")

                        metrics = benchmark_optimization(n_cities, n_vehicles, args.location,
                                                      backend, hybrid)
                        metrics['backend'] = backend
                        metrics['hybrid'] = hybrid
                        all_metrics.append(metrics)

                        logger.info("Performance metrics:")
                        for key, value in metrics.items():
                            if isinstance(value, (int, float)) and not isinstance(value, bool):
                                logger.info(f"{key}: {value:.3f}")
                            elif isinstance(value, dict):
                                logger.info(f"{key}: {value}")
                            else:
                                logger.info(f"{key}: {value}")

                        if metrics:
                            if 'network' in metrics and 'nodes' in metrics:
                                routes = metrics.get('routes', [])
                                if routes:
                                    node_routes = [[metrics['nodes'][i] for i in route] for route in routes]
                                    metrics['network'].create_folium_map(
                                        node_routes,
                                        save_path=f"route_map_{backend}_{'hybrid' if hybrid else 'pure'}_{n_cities}cities.html"
                                    )
                                    metrics['network'].create_static_map(
                                        node_routes,
                                        save_path=f"route_map_{backend}_{'hybrid' if hybrid else 'pure'}_{n_cities}cities.png"
                                    )

            visualizer.plot_benchmark_results(all_metrics, save_path="benchmark_results.png")
            visualizer.plot_time_series_analysis(all_metrics, save_path="time_series_analysis.png")
            visualizer.plot_cross_validation_metrics(all_metrics, save_path="cross_validation_metrics.png")
            logger.info("\nBenchmark visualizations have been saved.")

            logger.info("\nSummary Statistics:")
            for backend in sorted(set(m['backend'] for m in all_metrics)):
                backend_metrics = [m for m in all_metrics if m['backend'] == backend]
                avg_gap = np.mean([m['quantum_classical_gap'] for m in backend_metrics])
                avg_time = np.mean([m['optimization_time'] for m in backend_metrics])
                logger.info(f"{backend.upper()}:")
                logger.info(f"  Average gap to classical: {avg_gap:.1%}")
                logger.info(f"  Average optimization time: {avg_time:.2f}s")

                if any(m.get('hybrid', False) for m in backend_metrics):
                    hybrid_metrics = [m for m in backend_metrics if m.get('hybrid', False)]
                    pure_metrics = [m for m in backend_metrics if not m.get('hybrid', False)]
                    hybrid_improvement = 1 - np.mean([m['quantum_classical_gap'] for m in hybrid_metrics]) / \
                                          np.mean([m['quantum_classical_gap'] for m in pure_metrics])
                    hybrid_speedup = np.mean([m['total_time'] for m in pure_metrics]) / \
                                   np.mean([m['total_time'] for m in hybrid_metrics])
                    logger.info(f"  Hybrid improvement: {hybrid_improvement:.1%}")
                    logger.info(f"  Hybrid speedup: {hybrid_speedup:.2f}x")

            logger.info("\nCross-backend comparison:")
            sizes = sorted(set(m['problem_size']['n_cities'] for m in all_metrics))
            for size in sizes:
                size_metrics = [m for m in all_metrics if m['problem_size']['n_cities'] == size]
                logger.info(f"\nProblem size: {size} cities")
                for backend in backends:
                    backend_size_metrics = [m for m in size_metrics if m['backend'] == backend]
                    if backend_size_metrics:
                        avg_gap = np.mean([m['quantum_classical_gap'] for m in backend_size_metrics])
                        avg_time = np.mean([m['total_time'] for m in backend_size_metrics])
                        avg_variance = np.mean([m.get('cost_variance', 0) for m in backend_size_metrics])
                        logger.info(f"  {backend}: gap={avg_gap:.1%}, time={avg_time:.2f}s, "
                                     f"variance={avg_variance:.2e}")

            return

        n_cities = args.cities
        grid_size = args.grid_size
        qaoa_depth = args.qaoa_depth
        n_vehicles = args.vehicles
        vehicle_capacity = [args.capacity] * n_vehicles

        n_qubits = n_cities * n_cities * n_vehicles
        if n_qubits > 16:
            logger.error(f"Problem size too large: {n_qubits} qubits required. Please reduce number of cities or vehicles.")
            return

        logger.info(f"Starting QAOA optimization for {n_cities} cities with {n_vehicles} vehicles")
        logger.info("Note: Larger problem sizes may require more optimization steps")

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
            coordinates, nodes, network = generate_street_network_cities(n_cities, args.location)
            logger.info("Using generated coordinates: %s", coordinates)

        demands = generate_random_demands(n_cities)
        logger.info("City demands: %s", demands)

        qubo = QUBOFormulation(n_cities, n_vehicles, vehicle_capacity)
        distance_matrix = np.zeros((n_cities, n_cities))
        paths_between_cities = {}

        for i in range(n_cities):
            for j in range(n_cities):
                if i != j:
                    distance, path = manhattan_distance_with_obstacles(coordinates[i], coordinates[j], set(), grid_size, network)
                    distance_matrix[i,j] = distance
                    paths_between_cities[(i,j)] = path
        logger.info("\nDistance matrix:\n%s", distance_matrix)

        logger.info("\nSolving with classical methods...")

        start_time = time.time()
        optimal_route, optimal_length = solve_tsp_brute_force(distance_matrix)
        brute_force_time = time.time() - start_time

        start_time = time.time()
        cw_routes, cw_length = clarke_wright_savings(distance_matrix, demands, 
                                                   depot_index=0, capacity=vehicle_capacity[0])
        cw_time = time.time() - start_time

        start_time = time.time()
        if args.hybrid:
            circuit = HybridOptimizer(n_qubits, depth=qaoa_depth, backend=args.backend)
            logger.info(f"Initialized hybrid optimizer with {args.backend} backend")
        else:
            if args.backend == 'qiskit':
                circuit = QiskitQAOA(n_qubits, depth=qaoa_depth)
            else:
                circuit = QAOACircuit(n_qubits, depth=qaoa_depth)
            logger.info(f"Initialized {args.backend} circuit with {n_qubits} qubits")

        logger.info("Creating QUBO matrix...")
        qubo_matrix = qubo.create_qubo_matrix(distance_matrix, demands=demands, penalty=2.0)

        cost_terms = []
        max_coeff = np.max(np.abs(qubo_matrix))
        threshold = max_coeff * 0.001
        logger.info(f"Max coefficient: {max_coeff:.6f}, Threshold: {threshold:.6f}")

        n_terms_added = 0
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                if abs(qubo_matrix[i, j]) > threshold:
                    cost_terms.append((float(qubo_matrix[i, j]), (i, j)))
                    n_terms_added += 1
                    logger.debug(f"Added cost term: ({i},{j}) with coefficient {qubo_matrix[i,j]:.6f}")

        logger.info("Generated %d cost terms from QUBO matrix", n_terms_added)
        if n_terms_added == 0:
            logger.error("No cost terms generated. QUBO matrix might be too sparse or threshold too high.")
            return

        logger.info("Starting QAOA optimization...")

        try:
            params, costs = circuit.optimize(cost_terms, steps=10)
            if args.backend == 'qiskit':
                measurements = circuit.get_expectation_values(params, cost_terms)
            else:
                measurements = circuit.circuit(params, cost_terms)

            binary_solution = decode_measurements(measurements, n_cities)
            routes = qubo.decode_solution(binary_solution)
            quantum_time = time.time() - start_time
        except Exception as opt_error:
            logger.error(f"Optimization error with {args.backend} backend: %s", str(opt_error))
            raise


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

        quantum_metrics = {
            "distance": total_route_length,
            "time": quantum_time
        }

        classical_metrics = {
            "distance": cw_length,
            "time": cw_time
        }

        logger.info("\nSolution comparison:")
        logger.info(f"Quantum solution length: {total_route_length:.2f} (time: {quantum_time:.2f}s)")
        logger.info(f"Clarke-Wright solution length: {cw_length:.2f} (time: {cw_time:.2f}s)")
        logger.info(f"Brute force optimal length: {optimal_length:.2f} (time: {brute_force_time:.2f}s)")

        quantum_gap = (total_route_length - optimal_length) / optimal_length
        classical_gap = (cw_length - optimal_length) / optimal_length
        logger.info(f"Quantum solution gap: {quantum_gap:.1%}")
        logger.info(f"Classical solution gap: {classical_gap:.1%}")

        visualizer = CircuitVisualizer()
        try:
            node_routes = [[nodes[i] for i in route] for route in routes]
            # Save both HTML and PNG versions
            network.create_folium_map(node_routes, save_path="quantum_route.html")
            network.create_static_map(node_routes, save_path="quantum_route.png")
            visualizer.plot_optimization_trajectory(costs, save_path="optimization_trajectory.png")
            visualizer.plot_solution_comparison(coordinates, routes[0], cw_routes[0], 
                                                quantum_metrics, classical_metrics, 
                                                save_path="solution_comparison.png")
            logger.info("Visualizations saved as 'quantum_route.html', 'quantum_route.png', "
                       "'optimization_trajectory.png', and 'solution_comparison.png'")
        except Exception as viz_error:
            logger.error("Visualization error: %s", str(viz_error))

    except Exception as e:
        logger.error("Error in main: %s", str(e), exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()