from qaoa_core import QAOACircuit
from qubo_formulation import QUBOFormulation
from circuit_visualization import CircuitVisualizer
from optimization import QAOAOptimizer
from utils import Utils
import numpy as np
import logging

# Update logging configuration
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    try:
        # Problem parameters - reduced size for better convergence
        n_cities = 3  # Reduced from 4 for faster optimization
        qaoa_depth = 1  # Reduced from 2 for simpler circuit

        logger.info("Generating random cities...")
        coordinates = Utils.generate_random_cities(n_cities)

        logger.info("Creating QUBO formulation...")
        qubo = QUBOFormulation(n_cities)
        distance_matrix = qubo.create_distance_matrix(coordinates)
        qubo_matrix = qubo.create_qubo_matrix(distance_matrix)

        # Initialize QAOA circuit with reduced parameters
        n_qubits = n_cities * n_cities
        logger.info(f"Initializing QAOA circuit with {n_qubits} qubits and depth {qaoa_depth}")
        circuit = QAOACircuit(n_qubits, depth=qaoa_depth)

        # Create cost Hamiltonian terms from QUBO matrix
        cost_terms = []
        for i in range(n_qubits):
            for j in range(i, n_qubits):
                if abs(qubo_matrix[i, j]) > 1e-10:  # Numerical threshold
                    cost_terms.append((float(qubo_matrix[i, j]), [f"Z{i}", f"Z{j}"]))

        logger.info(f"Created {len(cost_terms)} cost Hamiltonian terms")

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
        except Exception as e:
            logger.error(f"Optimization failed: {str(e)}")
            raise

        # Get final measurements
        logger.info("Getting final measurements...")
        try:
            final_state = circuit.circuit(optimal_params, cost_terms)
            logger.info("Final measurements obtained successfully")
        except Exception as e:
            logger.error(f"Error getting final measurements: {str(e)}")
            raise

        # Convert continuous results to binary solution
        binary_solution = [1 if x > 0 else 0 for x in final_state]
        logger.info(f"Binary solution found: {binary_solution}")

        # Decode solution to get route
        logger.info("Decoding solution...")
        route = qubo.decode_solution(binary_solution)
        logger.info(f"Decoded route: {route}")

        # Verify and visualize solution
        if Utils.verify_solution(route, n_cities):
            logger.info("Valid solution found!")
            print(f"Optimal route: {route}")

            # Calculate route length
            route_length = Utils.calculate_route_length(route, distance_matrix)
            print(f"Route length: {route_length:.2f}")

            # Visualize results
            logger.info("Generating visualizations...")
            try:
                visualizer.plot_circuit_results(final_state, "Final Quantum State")
                visualizer.plot_optimization_trajectory(cost_history)
                visualizer.plot_route(coordinates, route)
                logger.info("Visualizations generated successfully")
            except Exception as e:
                logger.error(f"Error generating visualizations: {str(e)}")
                raise
        else:
            logger.warning("Invalid solution found. Try adjusting parameters.")
            print("Invalid solution found. Try adjusting parameters.")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()