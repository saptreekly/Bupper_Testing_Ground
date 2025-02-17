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
        logger.info("Number of cost terms: %d", len(cost_terms))

        # Initialize QAOA circuit
        circuit = QAOACircuit(4, depth=1)  # 4 qubits for 2 cities

        # Test circuit with fixed parameters
        test_params = np.array([np.pi/4, np.pi/2])  # Fixed test parameters
        measurements = circuit.circuit(test_params, cost_terms)
        logger.info("Test measurements shape: %d", len(measurements))
        assert len(measurements) == 4, f"Expected 4 measurements, got {len(measurements)}"

        # Run optimization with careful monitoring
        logger.info("Starting optimization")
        params, costs = circuit.optimize(cost_terms, steps=50)  # Increased steps for better convergence
        logger.info("Optimization completed")
        logger.info("Final parameters: %s", params)
        logger.info("Final cost: %.6f", costs[-1])

        # Plot optimization trajectory
        plt.figure(figsize=(10, 6))
        plt.plot(costs, 'b-')
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.title('QAOA Optimization Trajectory')
        plt.grid(True)
        plt.savefig('optimization_trajectory.png')
        plt.close()

        # Verify optimization progress
        assert len(costs) > 0, "No cost history recorded"
        assert costs[-1] <= costs[0], "Cost did not decrease during optimization"

        # Get final solution
        measurements = circuit.circuit(params, cost_terms)
        binary_solution = [1 if x > 0 else 0 for x in measurements]
        route = qubo.decode_solution(binary_solution)

        # Verify solution properties
        assert Utils.verify_solution(route, n_cities), "Invalid route found"
        route_length = Utils.calculate_route_length(route, distance_matrix)
        logger.info("Final route: %s, length: %.3f", route, route_length)

        # Verify route length is reasonable (should be close to 1.0 for this simple case)
        assert 0.9 <= route_length <= 1.1, f"Unexpected route length: {route_length}"

        logger.info("All tests passed successfully!")
        return True

    except Exception as e:
        logger.error("Test failed: %s", str(e), exc_info=True)
        raise

if __name__ == "__main__":
    test_minimal_qaoa()