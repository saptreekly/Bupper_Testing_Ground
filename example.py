from qaoa_core import QAOACircuit
from qubo_formulation import QUBOFormulation
from circuit_visualization import CircuitVisualizer
from optimization import QAOAOptimizer
from utils import Utils
import numpy as np
import logging
from typing import List, Tuple

# Update logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    try:
        # Problem parameters - reduced size for better convergence
        n_cities = 3  # Reduced from 4 for faster optimization
        qaoa_depth = 1  # Reduced from 2 for simpler circuit

        logger.info("Starting QAOA optimization for TSP")
        logger.info("Configuration: %d cities, QAOA depth %d", n_cities, qaoa_depth)

        logger.info("Generating random cities...")
        coordinates = Utils.generate_random_cities(n_cities)
        logger.info("Generated coordinates: %s", str(coordinates))

        logger.info("Creating QUBO formulation...")
        qubo = QUBOFormulation(n_cities)
        distance_matrix = qubo.create_distance_matrix(coordinates)
        logger.info("Distance matrix:\n%s", str(distance_matrix))
        qubo_matrix = qubo.create_qubo_matrix(distance_matrix)
        logger.info("QUBO matrix shape: %s", qubo_matrix.shape)

        # Initialize QAOA circuit with reduced parameters
        n_qubits = n_cities * n_cities
        logger.info("Initializing QAOA circuit with %d qubits and depth %d", n_qubits, qaoa_depth)
        circuit = QAOACircuit(n_qubits, depth=qaoa_depth)

        # Create cost Hamiltonian terms from QUBO matrix
        cost_terms = []
        for i in range(n_qubits):
            for j in range(i, n_qubits):
                if abs(qubo_matrix[i, j]) > 1e-10:  # Numerical threshold
                    cost_terms.append((float(qubo_matrix[i, j]), (i, j)))  # Using integer indices

        logger.info("Created %d cost Hamiltonian terms", len(cost_terms))

        # Initialize optimizer with reduced iterations
        logger.info("Initializing optimizer and visualizer...")
        optimizer = QAOAOptimizer(circuit.circuit, 2 * qaoa_depth)
        visualizer = CircuitVisualizer()

        # Optimize circuit parameters with fewer iterations
        logger.info("Starting optimization process...")
        try:
            optimal_params, cost_history = optimizer.optimize(
                cost_terms,
                max_iterations=20,  # Reduced from 50
                learning_rate=0.01,  # Reduced from 0.05
                tolerance=1e-3  # Increased from 1e-4
            )
            logger.info("Optimization completed successfully")
            logger.info("Final cost history: %s", str(cost_history))
        except Exception as e:
            logger.error("Optimization failed: %s", str(e))
            raise

        # Get final measurements
        logger.info("Getting final measurements...")
        try:
            measurements = circuit.circuit(optimal_params, cost_terms)
            logger.info("Final measurements shape: %d", len(measurements))
            logger.info("Measurements: %s", str(measurements))

            # Convert measurements to binary solution (threshold at 0)
            binary_solution = [1 if float(x) > 0 else 0 for x in measurements]
            logger.info("Binary solution: %s", str(binary_solution))

            # Decode solution to get route
            logger.info("Decoding solution...")
            route = qubo.decode_solution(binary_solution)
            logger.info("Decoded route: %s", str(route))

            # Verify and visualize solution
            if Utils.verify_solution(route, n_cities):
                logger.info("Valid solution found!")
                print(f"Optimal route: {route}")

                # Calculate route length
                route_length = Utils.calculate_route_length(route, distance_matrix)
                print(f"Route length: {route_length:.2f}")
                logger.info("Route length: %.2f", route_length)

                # Visualize results
                logger.info("Generating visualizations...")
                try:
                    visualizer.plot_circuit_results(measurements, "Final Quantum State") # Modified to use measurements
                    visualizer.plot_optimization_trajectory(cost_history)
                    visualizer.plot_route(coordinates, route)
                    logger.info("Visualizations generated successfully")
                except Exception as e:
                    logger.error("Error generating visualizations: %s", str(e))
                    raise
            else:
                logger.warning("Invalid solution found. Try adjusting parameters.")
                print("Invalid solution found. Try adjusting parameters.")

        except Exception as e:
            logger.error("Error getting final measurements: %s", str(e))
            raise

    except Exception as e:
        logger.error("An error occurred: %s", str(e), exc_info=True)
        raise

if __name__ == "__main__":
    main()