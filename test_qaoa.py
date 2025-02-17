import unittest
import numpy as np
import logging
from qaoa_core import QAOACircuit
from qubo_formulation import QUBOFormulation
from utils import Utils
from example import parse_coordinates, validate_coordinates
from hybrid_optimizer import HybridOptimizer
import matplotlib.pyplot as plt
from time import time

logging.basicConfig(level=logging.DEBUG,
                   format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

class TestQAOA(unittest.TestCase):
    """Test suite for QAOA implementation."""

    def setUp(self):
        """Set up test fixtures."""
        self.n_cities_small = 3
        self.n_cities_medium = 4
        self.grid_size = 8
        self.n_vehicles = 1
        self.vehicle_capacity = [float('inf')]

    def test_coordinate_parsing(self):
        """Test coordinate string parsing functionality."""
        try:
            # Test valid input
            valid_input = "0,0;2,2;4,4"
            coords = parse_coordinates(valid_input)
            self.assertEqual(len(coords), 3, f"Expected 3 coordinates, got {len(coords)}")
            self.assertEqual(coords, [(0,0), (2,2), (4,4)], "Incorrect coordinate parsing")
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
                self.assertEqual(coords, [], f"Expected empty list for invalid input: {invalid_input}")
            logger.info("Invalid coordinate parsing tests passed")

        except Exception as e:
            logger.error("Coordinate parsing test failed: %s", str(e))
            raise

    def test_coordinate_validation(self):
        """Test coordinate validation functionality."""
        try:
            # Test valid coordinates
            valid_coords = [(0,0), (2,2), (4,4)]
            self.assertTrue(validate_coordinates(valid_coords, self.grid_size), 
                          "Valid coordinates failed validation")

            # Test invalid coordinates
            invalid_coords = [
                [(0,0), (8,8), (2,2)],  # Out of bounds
                [(0,0), (0,0), (2,2)],  # Duplicate
                [(0,0), (-1,2), (2,2)], # Negative
                [(0,0), (2,9), (2,2)],  # Beyond grid
            ]

            for coords in invalid_coords:
                self.assertFalse(validate_coordinates(coords, self.grid_size),
                               f"Invalid coordinates passed validation: {coords}")

            logger.info("Coordinate validation tests passed")

        except Exception as e:
            logger.error("Coordinate validation test failed: %s", str(e))
            raise

    def test_minimal_qaoa(self):
        """Test QAOA with minimal 2-city problem."""
        try:
            # Setup minimal problem
            n_cities = 2
            coordinates = [(0.0, 0.0), (1.0, 0.0)]  # Two cities on x-axis, distance = 1
            logger.info("Testing with coordinates: %s", coordinates)

            # Create and validate QUBO
            qubo = QUBOFormulation(n_cities, self.n_vehicles, self.vehicle_capacity)
            distance_matrix = qubo.create_distance_matrix(coordinates)

            # Basic property tests
            self.assertTrue(np.allclose(distance_matrix, distance_matrix.T), 
                          "Distance matrix not symmetric")
            self.assertTrue(np.allclose(np.diag(distance_matrix), 0), 
                          "Diagonal elements not zero")
            self.assertTrue(np.allclose(distance_matrix[0,1], 1.0), 
                          "Incorrect distance")

            # Create QUBO matrix with simplified demands
            demands = [0.0, 1.0]  # Depot has no demand
            qubo_matrix = qubo.create_qubo_matrix(distance_matrix, demands=demands)

            # Verify QUBO properties
            n_qubits = n_cities * n_cities * self.n_vehicles
            self.assertEqual(qubo_matrix.shape, (n_qubits, n_qubits), 
                           f"Unexpected QUBO matrix shape: {qubo_matrix.shape}")
            self.assertTrue(np.allclose(qubo_matrix, qubo_matrix.T), 
                          "QUBO matrix not symmetric")

            # Test QAOA optimization
            circuit = QAOACircuit(n_qubits, depth=1)
            params, costs = circuit.optimize(self._create_cost_terms(qubo_matrix), steps=50)

            # Verify optimization results
            self.assertIsNotNone(params, "Optimization failed to return parameters")
            self.assertTrue(len(costs) > 0, "No optimization history recorded")

            # Check solution quality
            measurements = circuit.circuit(params, self._create_cost_terms(qubo_matrix))
            binary_solution = [1 if x > 0 else 0 for x in measurements]
            routes = qubo.decode_solution(binary_solution)

            for route in routes:
                self.assertEqual(route[0], 0, "Route doesn't start at depot")
                self.assertEqual(route[-1], 0, "Route doesn't end at depot")
                route_length = sum(distance_matrix[route[i], route[i+1]] 
                                 for i in range(len(route)-1))
                self.assertAlmostEqual(route_length, 2.0, delta=0.1, 
                                     msg=f"Unexpected route length: {route_length}")

            logger.info("Minimal QAOA test passed")

        except Exception as e:
            logger.error("Test failed: %s", str(e), exc_info=True)
            raise

    def test_hybrid_optimization(self):
        """Test hybrid quantum-classical optimization."""
        try:
            # Test both backends
            backends = ['qiskit', 'pennylane']
            for backend in backends:
                # Setup small problem
                coordinates = [(0,0), (1,0), (0,1)]  # Triangle configuration
                qubo = QUBOFormulation(self.n_cities_small, self.n_vehicles, self.vehicle_capacity)
                distance_matrix = qubo.create_distance_matrix(coordinates)
                demands = [0.0] + [1.0] * (self.n_cities_small - 1)

                # Create QUBO matrix
                qubo_matrix = qubo.create_qubo_matrix(distance_matrix, demands=demands)
                n_qubits = self.n_cities_small * self.n_cities_small * self.n_vehicles

                # Initialize hybrid optimizer
                optimizer = HybridOptimizer(n_qubits, depth=1, backend=backend)

                # Run optimization
                start_time = time()
                params, costs = optimizer.optimize(self._create_cost_terms(qubo_matrix), steps=30)
                end_time = time()

                # Verify results
                self.assertIsNotNone(params, f"Hybrid optimization failed with {backend}")
                self.assertTrue(len(costs) > 0, f"No optimization history for {backend}")

                # Performance check
                runtime = end_time - start_time
                logger.info(f"Hybrid optimization with {backend} completed in {runtime:.2f}s")

                # Solution quality check
                measurements = optimizer.get_expectation_values(params, self._create_cost_terms(qubo_matrix))
                binary_solution = [1 if x > 0 else 0 for x in measurements]
                routes = qubo.decode_solution(binary_solution)

                # Verify route properties
                for route in routes:
                    self.assertEqual(route[0], 0, f"Invalid route start with {backend}")
                    self.assertEqual(route[-1], 0, f"Invalid route end with {backend}")
                    route_length = sum(distance_matrix[route[i], route[i+1]] 
                                     for i in range(len(route)-1))
                    self.assertGreater(route_length, 0, f"Invalid route length with {backend}")

            logger.info("Hybrid optimization tests passed")

        except Exception as e:
            logger.error("Hybrid optimization test failed: %s", str(e))
            raise

    def _create_cost_terms(self, qubo_matrix: np.ndarray) -> list:
        """Helper method to create cost terms from QUBO matrix."""
        cost_terms = []
        n_qubits = qubo_matrix.shape[0]
        for i in range(n_qubits):
            for j in range(i, n_qubits):
                if abs(qubo_matrix[i, j]) > 1e-10:
                    cost_terms.append((float(qubo_matrix[i, j]), (i, j)))
        return cost_terms

if __name__ == "__main__":
    unittest.main()