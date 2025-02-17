import numpy as np
import logging
from qaoa_core import QAOACircuit
from qubo_formulation import QUBOFormulation
from utils import Utils
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.DEBUG,
                   format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

def test_minimal_qaoa():
    """Test QAOA with minimal 2-city problem."""
    try:
        # Setup minimal problem
        n_cities = 2
        coordinates = [(0.0, 0.0), (1.0, 0.0)]  # Two cities on x-axis, distance = 1
        logger.info("Testing with coordinates: %s", coordinates)

        # Create and validate QUBO
        qubo = QUBOFormulation(n_cities)
        distance_matrix = qubo.create_distance_matrix(coordinates)
        logger.info("Distance matrix:\n%s", distance_matrix)

        # Verify distance matrix properties
        assert np.allclose(distance_matrix, distance_matrix.T), "Distance matrix not symmetric"
        assert np.allclose(np.diag(distance_matrix), 0), "Diagonal elements not zero"
        assert np.allclose(distance_matrix[0,1], 1.0), "Incorrect distance"

        # Create QUBO matrix
        qubo_matrix = qubo.create_qubo_matrix(distance_matrix)
        logger.info("QUBO matrix:\n%s", qubo_matrix)

        # Verify QUBO matrix properties
        assert qubo_matrix.shape == (4, 4), f"Unexpected QUBO matrix shape: {qubo_matrix.shape}"
        assert np.allclose(qubo_matrix, qubo_matrix.T), "QUBO matrix not symmetric"

        # Create simplified cost terms
        cost_terms = []
        for i in range(4):
            for j in range(i, 4):
                if abs(qubo_matrix[i, j]) > 1e-10:
                    cost_terms.append((float(qubo_matrix[i, j]), (i, j)))
        logger.info("Cost terms: %s", cost_terms)

        # Initialize QAOA circuit
        circuit = QAOACircuit(4, depth=1)  # 4 qubits for 2 cities
        logger.info("Initialized QAOA circuit with 4 qubits")

        # Run optimization with careful monitoring
        logger.info("Starting optimization")
        params, costs = circuit.optimize(cost_terms, steps=50)
        logger.info("Optimization completed")
        logger.info("Final parameters: %s", params)
        logger.info("Final cost: %.6f", costs[-1])

        # Get final solution
        measurements = circuit.circuit(params, cost_terms)
        logger.debug("Final measurements: %s", measurements)

        binary_solution = [1 if x > 0 else 0 for x in measurements]
        logger.debug("Binary solution: %s", binary_solution)

        route = qubo.decode_solution(binary_solution)
        logger.info("Decoded route: %s", route)

        # Verify solution validity
        assert Utils.verify_solution(route, n_cities), "Invalid route found"

        # Calculate and verify route length
        route_length = Utils.calculate_route_length(route, distance_matrix)
        logger.info("Route length calculation:")
        logger.info("- Route: %s", route)
        logger.info("- Distance matrix:\n%s", distance_matrix)
        logger.info("- Final route length: %.3f", route_length)

        # Verify route length is reasonable (should be close to 1.0 for this simple case)
        assert 0.9 <= route_length <= 1.1, f"Unexpected route length: {route_length}"

        logger.info("All tests passed successfully!")
        return True

    except Exception as e:
        logger.error("Test failed: %s", str(e), exc_info=True)
        raise

if __name__ == "__main__":
    test_minimal_qaoa()