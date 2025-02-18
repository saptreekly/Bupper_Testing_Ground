import unittest
import numpy as np
import logging
from qaoa_core import QAOACircuit
from qiskit_qaoa import QiskitQAOA
from qubo_formulation import QUBOFormulation
from utils import Utils
from example import parse_coordinates, validate_coordinates
import matplotlib.pyplot as plt
from time import time

logging.basicConfig(level=logging.DEBUG,
                   format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

class TestQAOA(unittest.TestCase):
    """Test suite for QAOA implementation with enhanced validation."""

    def setUp(self):
        """Set up test fixtures."""
        self.n_cities_small = 3
        self.n_cities_medium = 4
        self.grid_size = 8
        self.n_vehicles = 1
        self.vehicle_capacity = [float('inf')]

    def test_qiskit_backend_initialization(self):
        """Test Qiskit backend initialization and parameter handling."""
        try:
            # Test initialization with different qubit counts
            test_sizes = [2, 3, 5]
            for n_qubits in test_sizes:
                logger.info(f"Testing initialization with {n_qubits} qubits")
                optimizer = QiskitQAOA(n_qubits)
                self.assertIsNotNone(optimizer.estimator, "Estimator not initialized")
                self.assertIsNotNone(optimizer.backend, "Backend not initialized")

            # Test parameter validation
            optimizer = QiskitQAOA(3)
            params = np.array([0.1, 0.2, 0.3])  # Extra parameter
            validated = optimizer._validate_and_truncate_params(params)
            self.assertEqual(len(validated), 2, "Parameters not truncated correctly")
            self.assertTrue(np.all(validated <= 2*np.pi), "Parameters not bounded correctly")

        except Exception as e:
            logger.error(f"Backend initialization test failed: {str(e)}")
            raise

    def test_circuit_creation(self):
        """Test quantum circuit creation and validation."""
        try:
            n_qubits = 3
            optimizer = QiskitQAOA(n_qubits)

            # Test with valid cost terms
            cost_terms = [
                (1.0, (0, 1)),
                (0.5, (1, 2))
            ]
            params = np.array([0.1, 0.2])

            circuit = optimizer._create_circuit(params, cost_terms)
            self.assertIsNotNone(circuit, "Circuit creation failed")
            self.assertEqual(circuit.num_qubits, n_qubits, "Incorrect number of qubits")

            # Test with invalid indices
            invalid_terms = [(1.0, (0, n_qubits))]  # Out of bounds
            circuit = optimizer._create_circuit(params, invalid_terms)
            self.assertIsNotNone(circuit, "Circuit should be created even with invalid terms")

            # Test parameter bounds
            extreme_params = np.array([10.0, -5.0])
            validated = optimizer._validate_and_truncate_params(extreme_params)
            self.assertTrue(np.all(np.abs(validated) <= 2*np.pi), "Parameters not properly bounded")

        except Exception as e:
            logger.error(f"Circuit creation test failed: {str(e)}")
            raise

    def test_expectation_values(self):
        """Test expectation value computation with detailed validation."""
        try:
            n_qubits = 2
            optimizer = QiskitQAOA(n_qubits)

            # Simple test case with known parameters
            cost_terms = [(1.0, (0, 1))]
            params = np.array([0.0, np.pi/4])  # Simple parameters

            measurements = optimizer.get_expectation_values(params, cost_terms)
            self.assertEqual(len(measurements), n_qubits, "Incorrect measurement size")
            self.assertTrue(np.all(np.abs(measurements) <= 1.0), "Invalid measurement values")

            # Test empty cost terms
            empty_measurements = optimizer.get_expectation_values(params, [])
            self.assertEqual(len(empty_measurements), n_qubits, "Incorrect handling of empty terms")

            # Log measurement statistics
            logger.info(f"Measurement results: {measurements}")
            logger.info(f"Measurement statistics - Mean: {np.mean(measurements):.4f}, Std: {np.std(measurements):.4f}")

        except Exception as e:
            logger.error(f"Expectation value test failed: {str(e)}")
            raise

    def test_partition_handling(self):
        """Test partition handling for different problem sizes."""
        try:
            # Test small problem (no partitioning needed)
            n_small = 3
            optimizer_small = QiskitQAOA(n_small)
            cost_terms_small = [(1.0, (0, 1)), (0.5, (1, 2))]

            # Test large problem (requires partitioning)
            n_large = 35
            optimizer_large = QiskitQAOA(n_large)

            # Create cost terms for large problem
            cost_terms_large = []
            for i in range(0, n_large-1, 2):
                cost_terms_large.append((1.0, (i, i+1)))

            # Test both cases
            params = np.array([0.1, 0.2])

            # Small case
            measurements_small = optimizer_small.get_expectation_values(params, cost_terms_small)
            self.assertEqual(len(measurements_small), n_small, "Incorrect small case measurements")

            # Large case
            measurements_large = optimizer_large.get_expectation_values(params, cost_terms_large)
            self.assertEqual(len(measurements_large), n_large, "Incorrect large case measurements")

            # Log partition statistics
            logger.info(f"Small case measurements shape: {measurements_small.shape}")
            logger.info(f"Large case measurements shape: {measurements_large.shape}")
            logger.info(f"Large case partition count: {optimizer_large.n_partitions}")

        except Exception as e:
            logger.error(f"Partition handling test failed: {str(e)}")
            raise

    def test_error_handling(self):
        """Test error handling and recovery."""
        try:
            optimizer = QiskitQAOA(3)

            # Test invalid parameters
            invalid_params = np.array([])
            validated = optimizer._validate_and_truncate_params(invalid_params)
            self.assertEqual(len(validated), 2, "Invalid parameter handling failed")

            # Test invalid cost terms
            invalid_terms = [(1.0, (0, 10))]  # Index out of bounds
            measurements = optimizer.get_expectation_values(np.array([0.1, 0.2]), invalid_terms)
            self.assertIsNotNone(measurements, "Error recovery failed")

            # Test extreme values
            extreme_params = np.array([1e6, -1e6])
            validated = optimizer._validate_and_truncate_params(extreme_params)
            self.assertTrue(np.all(np.isfinite(validated)), "Extreme value handling failed")

        except Exception as e:
            logger.error(f"Error handling test failed: {str(e)}")
            raise

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
            measurements = circuit.get_expectation_values(params, self._create_cost_terms(qubo_matrix))
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
                logger.info(f"Testing hybrid optimization with {backend} backend")

                # Setup small problem
                coordinates = [(0,0), (1,0), (0,1)]  # Triangle configuration
                qubo = QUBOFormulation(self.n_cities_small, self.n_vehicles, self.vehicle_capacity)
                distance_matrix = qubo.create_distance_matrix(coordinates)
                demands = [0.0] + [1.0] * (self.n_cities_small - 1)

                # Create QUBO matrix
                qubo_matrix = qubo.create_qubo_matrix(distance_matrix, demands=demands)
                n_qubits = self.n_cities_small * self.n_cities_small * self.n_vehicles

                # Initialize hybrid optimizer with error handling
                try:
                    optimizer = HybridOptimizer(n_qubits, depth=1, backend=backend)
                except Exception as init_error:
                    logger.error(f"Failed to initialize {backend} optimizer: {str(init_error)}")
                    continue

                # Run optimization with timing
                start_time = time()
                try:
                    params, costs = optimizer.optimize(self._create_cost_terms(qubo_matrix), steps=30)
                    end_time = time()
                    runtime = end_time - start_time

                    # Log optimization statistics
                    logger.info(f"Hybrid optimization with {backend} completed in {runtime:.2f}s")
                    if len(costs) > 0:
                        logger.info(f"Final cost: {costs[-1]:.6f}")
                        logger.info(f"Cost improvement: {(costs[0] - costs[-1])/abs(costs[0]):.1%}")

                    # Verify results
                    self.assertIsNotNone(params, f"Hybrid optimization failed with {backend}")
                    self.assertTrue(len(costs) > 0, f"No optimization history for {backend}")

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
                        logger.info(f"Route length with {backend}: {route_length:.2f}")

                except Exception as opt_error:
                    logger.error(f"Optimization failed for {backend}: {str(opt_error)}")
                    raise

            logger.info("Hybrid optimization tests completed successfully")

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