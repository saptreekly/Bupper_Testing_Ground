import logging
from qaoa_core import QAOACircuit
from qubo_formulation import QUBOFormulation
from circuit_visualization import CircuitVisualizer
from utils import Utils
import numpy as np

logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG for more detailed logs
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    try:
        # Increase to 3 cities with higher QAOA depth
        n_cities = 3
        qaoa_depth = 2  # Increased depth for better optimization

        logger.info("Starting QAOA optimization for TSP")
        logger.info("Configuration: %d cities, QAOA depth %d", n_cities, qaoa_depth)

        # Generate triangle test case
        coordinates = [
            (0.0, 0.0),  # First city at origin
            (1.0, 0.0),  # Second city at (1,0)
            (0.5, 0.866)  # Third city at equilateral triangle point
        ]
        logger.info("Test coordinates: %s", str(coordinates))

        # Create QUBO formulation
        qubo = QUBOFormulation(n_cities)
        distance_matrix = qubo.create_distance_matrix(coordinates)
        logger.info("Distance matrix:\n%s", str(distance_matrix))
        qubo_matrix = qubo.create_qubo_matrix(distance_matrix, penalty=4.0)  # Increased penalty
        logger.info("QUBO matrix:\n%s", str(qubo_matrix))

        # Initialize circuit with more qubits
        n_qubits = n_cities * n_cities  # Now 9 qubits for 3 cities
        circuit = QAOACircuit(n_qubits, depth=qaoa_depth)

        # Create cost terms
        cost_terms = []
        for i in range(n_qubits):
            for j in range(i, n_qubits):
                if abs(qubo_matrix[i, j]) > 1e-10:
                    cost_terms.append((float(qubo_matrix[i, j]), (i, j)))
        logger.info("Number of cost terms: %d", len(cost_terms))

        # Run optimization with more iterations for the larger problem
        params, cost_history = circuit.optimize(cost_terms, steps=200)  # Increased steps
        logger.info("Optimization completed")
        logger.info("Final parameters: %s", str(params))
        logger.info("Cost history: %s", str(cost_history))

        # Get final measurements and decode solution
        measurements = circuit.circuit(params, cost_terms)
        logger.info("Final measurements: %s", str(measurements))

        binary_solution = [1 if x > 0 else 0 for x in measurements]
        route = qubo.decode_solution(binary_solution)
        logger.info("Final route: %s", str(route))

        if Utils.verify_solution(route, n_cities):
            route_length = Utils.calculate_route_length(route, distance_matrix)
            logger.info("Valid solution found!")
            logger.info("Route: %s, Length: %.3f", str(route), route_length)

            # Visualize the route
            visualizer = CircuitVisualizer()
            visualizer.plot_route(coordinates, route)
            visualizer.plot_optimization_trajectory(cost_history)
        else:
            logger.warning("Invalid solution found")

    except Exception as e:
        logger.error("Error in main: %s", str(e), exc_info=True)
        raise

if __name__ == "__main__":
    main()