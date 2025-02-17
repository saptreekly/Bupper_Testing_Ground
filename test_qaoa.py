import unittest
import numpy as np
import logging
from qaoa_core import QAOACircuit
from qubo_formulation import QUBOFormulation
from utils import Utils
from example import parse_coordinates, validate_coordinates
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.DEBUG,
                   format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

def test_coordinate_parsing():
    """Test coordinate string parsing functionality."""
    try:
        # Test valid input
        valid_input = "0,0;2,2;4,4"
        coords = parse_coordinates(valid_input)
        assert len(coords) == 3, f"Expected 3 coordinates, got {len(coords)}"
        assert coords == [(0,0), (2,2), (4,4)], "Incorrect coordinate parsing"
        logger.info("Valid coordinate parsing test passed")

        # Test invalid inputs
        invalid_inputs = [
            "0,0,0;1,1",  # Wrong format
            "a,b;2,2",    # Non-numeric
            "0,0;",       # Incomplete
            "",          # Empty
            "0,0;1;1",   # Malformed
        ]

        for invalid_input in invalid_inputs:
            coords = parse_coordinates(invalid_input)
            assert coords == [], f"Expected empty list for invalid input: {invalid_input}"
        logger.info("Invalid coordinate parsing tests passed")

    except Exception as e:
        logger.error("Coordinate parsing test failed: %s", str(e))
        raise

def test_coordinate_validation():
    """Test coordinate validation functionality."""
    try:
        grid_size = 8

        # Test valid coordinates
        valid_coords = [(0,0), (2,2), (4,4)]
        assert validate_coordinates(valid_coords, grid_size), "Valid coordinates failed validation"

        # Test invalid coordinates
        invalid_coords = [
            [(0,0), (8,8), (2,2)],  # Out of bounds
            [(0,0), (0,0), (2,2)],  # Duplicate
            [(0,0), (-1,2), (2,2)], # Negative
            [(0,0), (2,9), (2,2)],  # Beyond grid
        ]

        for coords in invalid_coords:
            assert not validate_coordinates(coords, grid_size), \
                   f"Invalid coordinates passed validation: {coords}"

        logger.info("Coordinate validation tests passed")

    except Exception as e:
        logger.error("Coordinate validation test failed: %s", str(e))
        raise

def test_minimal_qaoa():
    """Test QAOA with minimal 2-city problem."""
    try:
        # Setup minimal problem
        n_cities = 2
        n_vehicles = 1
        vehicle_capacity = [float('inf')]
        coordinates = [(0.0, 0.0), (1.0, 0.0)]  # Two cities on x-axis, distance = 1
        logger.info("Testing with coordinates: %s", coordinates)

        # Create and validate QUBO
        qubo = QUBOFormulation(n_cities, n_vehicles, vehicle_capacity)
        distance_matrix = qubo.create_distance_matrix(coordinates)
        logger.info("Distance matrix:\n%s", distance_matrix)

        # Verify distance matrix properties
        assert np.allclose(distance_matrix, distance_matrix.T), "Distance matrix not symmetric"
        assert np.allclose(np.diag(distance_matrix), 0), "Diagonal elements not zero"
        assert np.allclose(distance_matrix[0,1], 1.0), "Incorrect distance"

        # Create QUBO matrix with simplified demands (all 1.0 except depot)
        demands = [0.0, 1.0]  # Depot has no demand
        qubo_matrix = qubo.create_qubo_matrix(distance_matrix, demands=demands)
        logger.info("QUBO matrix:\n%s", qubo_matrix)

        # Verify QUBO matrix properties
        n_qubits = n_cities * n_cities * n_vehicles
        assert qubo_matrix.shape == (n_qubits, n_qubits), f"Unexpected QUBO matrix shape: {qubo_matrix.shape}"
        assert np.allclose(qubo_matrix, qubo_matrix.T), "QUBO matrix not symmetric"

        # Create simplified cost terms
        cost_terms = []
        for i in range(n_qubits):
            for j in range(i, n_qubits):
                if abs(qubo_matrix[i, j]) > 1e-10:
                    cost_terms.append((float(qubo_matrix[i, j]), (i, j)))
        logger.info("Cost terms: %s", cost_terms)

        # Initialize QAOA circuit
        circuit = QAOACircuit(n_qubits, depth=1)
        logger.info("Initialized QAOA circuit with %d qubits", n_qubits)

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

        routes = qubo.decode_solution(binary_solution)
        logger.info("Decoded routes: %s", routes)

        for route in routes:
            # Verify each route starts and ends at depot (0)
            assert route[0] == 0 and route[-1] == 0, "Route doesn't start/end at depot"

            # Calculate and verify route length
            route_length = sum(distance_matrix[route[i], route[i+1]] 
                             for i in range(len(route)-1))
            logger.info("Route length: %.3f", route_length)

            # For this simple case, route should be [0, 1, 0] with length 2.0
            assert 1.9 <= route_length <= 2.1, f"Unexpected route length: {route_length}"

        logger.info("All tests passed successfully!")
        return True

    except Exception as e:
        logger.error("Test failed: %s", str(e), exc_info=True)
        raise

if __name__ == "__main__":
    test_coordinate_parsing()
    test_coordinate_validation()
    test_minimal_qaoa()