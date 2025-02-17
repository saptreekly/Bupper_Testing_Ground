from qaoa_core import QAOACircuit
from qubo_formulation import QUBOFormulation
from circuit_visualization import CircuitVisualizer
from utils import Utils
import numpy as np
import logging
from typing import List, Tuple

logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG for more detailed logs
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    try:
        # Minimal problem size for testing
        n_cities = 2  # Start with just 2 cities
        qaoa_depth = 1

        logger.info("Starting QAOA optimization for TSP")
        logger.info("Configuration: %d cities, QAOA depth %d", n_cities, qaoa_depth)

        # Generate simple test case
        coordinates = [(0.0, 0.0), (1.0, 0.0)]  # Two cities on x-axis
        logger.info("Test coordinates: %s", str(coordinates))

        # Create QUBO formulation
        qubo = QUBOFormulation(n_cities)
        distance_matrix = qubo.create_distance_matrix(coordinates)
        logger.info("Distance matrix:\n%s", str(distance_matrix))
        qubo_matrix = qubo.create_qubo_matrix(distance_matrix)
        logger.info("QUBO matrix:\n%s", str(qubo_matrix))

        # Initialize circuit
        n_qubits = n_cities * n_cities
        circuit = QAOACircuit(n_qubits, depth=qaoa_depth)

        # Create simplified cost terms
        cost_terms = []
        for i in range(n_qubits):
            for j in range(i, n_qubits):
                if abs(qubo_matrix[i, j]) > 1e-10:
                    cost_terms.append((float(qubo_matrix[i, j]), (i, j)))
        logger.info("Cost terms: %s", str(cost_terms))

        # Run optimization
        params, cost_history = circuit.optimize(cost_terms, steps=20)
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
            visualizer = CircuitVisualizer()
            visualizer.plot_route(coordinates, route)
        else:
            logger.warning("Invalid solution found")

    except Exception as e:
        logger.error("Error in main: %s", str(e), exc_info=True)
        raise

if __name__ == "__main__":
    main()