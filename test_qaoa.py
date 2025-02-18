import unittest
import numpy as np
import logging
from qaoa_core import QAOACircuit
from qubo_formulation import QUBOFormulation
from utils import Utils
from example import parse_coordinates, validate_coordinates
import matplotlib.pyplot as plt
import pennylane as qml # Added import for PennyLane


logging.basicConfig(level=logging.DEBUG,
                   format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

class TestQAOA(unittest.TestCase):
    """Test suite for QAOA implementation with enhanced gradient and parameter validation testing."""

    def setUp(self):
        """Set up test fixtures."""
        self.n_cities_small = 3
        self.n_cities_medium = 4
        self.grid_size = 8
        self.n_vehicles = 1
        self.vehicle_capacity = [float('inf')]
        self.test_params = np.array([0.1, 0.2])

    def test_circuit_initialization(self):
        """Test quantum circuit initialization with different configurations."""
        try:
            # Test initialization with different qubit counts
            test_sizes = [2, 3, 5]
            for n_qubits in test_sizes:
                logger.info(f"Testing initialization with {n_qubits} qubits")
                circuit = QAOACircuit(n_qubits)
                self.assertEqual(len(circuit.devices), circuit.n_partitions)
                self.assertEqual(len(circuit.circuits), circuit.n_partitions)

                # Verify device parameters
                for dev in circuit.devices:
                    self.assertIsNone(dev.shots)  # Ensure analytic mode
                    self.assertTrue(hasattr(dev, 'wires'))  # Check for device attributes

        except Exception as e:
            logger.error(f"Circuit initialization test failed: {str(e)}")
            raise

    def test_gradient_computation(self):
        """Test gradient computation with error handling."""
        try:
            n_qubits = 2
            circuit = QAOACircuit(n_qubits)

            # Test with simple cost terms
            cost_terms = [(1.0, (0, 1))]

            # Test gradient computation
            params = self.test_params.copy()
            measurements = circuit.get_expectation_values(params, cost_terms)

            # Verify measurements shape and bounds
            self.assertEqual(len(measurements), n_qubits)
            self.assertTrue(np.all(np.abs(measurements) <= 1.0))

            # Test optimization step
            new_params, costs = circuit.optimize(cost_terms, steps=1)
            self.assertEqual(len(new_params), 2)
            self.assertTrue(len(costs) > 0)

            # Verify parameter bounds
            self.assertTrue(np.all(np.abs(new_params) <= 2*np.pi))

        except Exception as e:
            logger.error(f"Gradient computation test failed: {str(e)}")
            raise

    def test_parameter_validation(self):
        """Test parameter validation with enhanced error checking."""
        try:
            circuit = QAOACircuit(3)

            # Test empty parameters
            empty_params = np.array([])
            validated = circuit._validate_and_truncate_params(empty_params)
            self.assertEqual(len(validated), 2)
            self.assertTrue(np.allclose(validated, np.array([0.01, np.pi/4])))  # Updated expected values

            # Test short parameters
            short_params = np.array([0.1])
            validated = circuit._validate_and_truncate_params(short_params)
            self.assertEqual(len(validated), 2)
            self.assertEqual(validated[0], 0.1)
            self.assertEqual(validated[1], 0.0)

            # Test long parameters
            long_params = np.array([0.1, 0.2, 0.3])
            validated = circuit._validate_and_truncate_params(long_params)
            self.assertEqual(len(validated), 2)
            self.assertEqual(validated[0], 0.1)
            self.assertEqual(validated[1], 0.2)

            # Test bounds
            extreme_params = np.array([10.0, -5.0])
            validated = circuit._validate_and_truncate_params(extreme_params)
            self.assertTrue(np.abs(validated[0]) <= 2*np.pi)
            self.assertTrue(np.abs(validated[1]) <= np.pi)

            logger.info("Parameter validation tests passed")

        except Exception as e:
            logger.error(f"Parameter validation test failed: {str(e)}")
            raise

    def test_expectation_values(self):
        """Test expectation value computation."""
        try:
            n_qubits = 2
            circuit = QAOACircuit(n_qubits)

            # Test with simple cost terms
            cost_terms = [(1.0, (0, 1))]
            params = np.array([0.0, np.pi/4])

            measurements = circuit.get_expectation_values(params, cost_terms)
            self.assertEqual(len(measurements), n_qubits)
            self.assertTrue(np.all(np.abs(measurements) <= 1.0))

            # Test empty cost terms
            empty_measurements = circuit.get_expectation_values(params, [])
            self.assertEqual(len(empty_measurements), n_qubits)

            # Log measurement statistics
            logger.info(f"Measurement results: {measurements}")
            logger.info(f"Measurement statistics - Mean: {np.mean(measurements):.4f}, "
                       f"Std: {np.std(measurements):.4f}")

        except Exception as e:
            logger.error(f"Expectation value test failed: {str(e)}")
            raise

    def test_partition_handling(self):
        """Test partition handling for different problem sizes with enhanced error checking."""
        try:
            # Test small problem (no partitioning needed)
            n_small = 3
            circuit_small = QAOACircuit(n_small)
            cost_terms_small = [(1.0, (0, 1)), (0.5, (1, 2))]

            # Test large problem (requires partitioning)
            n_large = 5  # Testing with 5 cities
            circuit_large = QAOACircuit(n_large)

            # Create cost terms for large problem
            cost_terms_large = [(1.0, (i, i+1)) for i in range(n_large-1)]

            # Test both cases
            params = self.test_params.copy()

            # Small case
            measurements_small = circuit_small.get_expectation_values(params, cost_terms_small)
            self.assertEqual(len(measurements_small), n_small)

            # Large case
            measurements_large = circuit_large.get_expectation_values(params, cost_terms_large)
            self.assertEqual(len(measurements_large), n_large)

            logger.info(f"Successfully tested partitioning for {n_large} qubits")

        except Exception as e:
            logger.error(f"Partition handling test failed: {str(e)}")
            raise

    def test_coordinate_parsing(self):
        """Test coordinate string parsing functionality."""
        try:
            # Test valid input
            valid_input = "0,0;2,2;4,4"
            coords = parse_coordinates(valid_input)
            self.assertEqual(len(coords), 3)
            self.assertEqual(coords, [(0,0), (2,2), (4,4)])
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
            self.assertTrue(validate_coordinates(valid_coords, self.grid_size))

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
            self.assertTrue(np.allclose(distance_matrix, distance_matrix.T))
            self.assertTrue(np.allclose(np.diag(distance_matrix), 0))
            self.assertTrue(np.allclose(distance_matrix[0,1], 1.0))

            # Create QUBO matrix with simplified demands
            demands = [0.0, 1.0]  # Depot has no demand
            qubo_matrix = qubo.create_qubo_matrix(distance_matrix, demands=demands)

            # Verify QUBO properties
            n_qubits = n_cities * n_cities * self.n_vehicles
            self.assertEqual(qubo_matrix.shape, (n_qubits, n_qubits))
            self.assertTrue(np.allclose(qubo_matrix, qubo_matrix.T))

            # Test QAOA optimization
            circuit = QAOACircuit(n_qubits, depth=1)
            params, costs = circuit.optimize(self._create_cost_terms(qubo_matrix), steps=2)

            # Verify optimization results
            self.assertIsNotNone(params)
            self.assertTrue(len(costs) > 0)

            logger.info("Minimal QAOA test passed")

        except Exception as e:
            logger.error(f"Test failed: {str(e)}")
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