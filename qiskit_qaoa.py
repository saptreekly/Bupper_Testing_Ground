import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import Aer
from qiskit.circuit import Parameter
from qiskit.primitives import BackendEstimator
from qiskit.quantum_info import SparsePauliOp
import logging

logger = logging.getLogger(__name__)

class QiskitQAOA:
    """QAOA implementation using Qiskit.

    This class provides a Qiskit-based implementation of the Quantum Approximate
    Optimization Algorithm (QAOA) for combinatorial optimization problems.

    Attributes:
        n_qubits (int): Number of qubits in the quantum circuit
        depth (int): Number of QAOA layers
        backend: Qiskit backend for simulation
        estimator: Qiskit primitive for expectation value estimation
    """

    def __init__(self, n_qubits: int, depth: int = 1):
        """Initialize QAOA circuit with Qiskit backend.

        Args:
            n_qubits: Number of qubits (limited to 25 for practical simulation)
            depth: Number of QAOA layers (default: 1)
        """
        self.n_qubits = min(n_qubits, 25)  # Hard limit at 25 qubits
        self.depth = depth
        try:
            self.backend = Aer.get_backend('aer_simulator')
            self.estimator = BackendEstimator(backend=self.backend)
            logger.info("Initialized Qiskit backend with %d qubits", self.n_qubits)
            logger.debug("Using simulator backend: %s", self.backend.backend_name)
        except Exception as e:
            logger.error("Failed to initialize Qiskit backend: %s", str(e))
            raise

    def _validate_cost_terms(self, cost_terms):
        """Validate and normalize cost terms.

        Args:
            cost_terms: List of (coefficient, (i, j)) tuples

        Returns:
            List of validated and normalized cost terms
        """
        if not cost_terms:
            logger.warning("Empty cost terms provided")
            return []

        valid_terms = []
        max_coeff = max(abs(coeff) for coeff, _ in cost_terms)

        if max_coeff < 1e-10:
            logger.warning("All coefficients are nearly zero")
            return []

        for coeff, (i, j) in cost_terms:
            if not (0 <= i < self.n_qubits and 0 <= j < self.n_qubits):
                logger.debug(f"Skipping invalid qubit indices: ({i}, {j})")
                continue
            if abs(coeff) > 1e-10:  # Only include significant terms
                norm_coeff = coeff / max_coeff
                valid_terms.append((norm_coeff, (i, j)))
                logger.debug(f"Added cost term: ({i}, {j}) with coefficient {norm_coeff:.6f}")

        return valid_terms

    def _create_qaoa_circuit(self, params, cost_terms):
        """Create QAOA circuit with given parameters.

        Args:
            params: Circuit parameters (gamma, beta for each layer)
            cost_terms: List of (coefficient, (i, j)) tuples

        Returns:
            QuantumCircuit: Constructed QAOA circuit
        """
        circuit = QuantumCircuit(self.n_qubits)
        valid_terms = self._validate_cost_terms(cost_terms)

        if not valid_terms:
            raise ValueError("No valid cost terms for circuit construction")

        # Initial state preparation
        circuit.h(range(self.n_qubits))
        logger.debug("Initialized circuit in uniform superposition state")

        # QAOA layers
        for p in range(self.depth):
            gamma = params[2 * p]
            beta = params[2 * p + 1]

            # Cost operator
            for coeff, (i, j) in valid_terms:
                circuit.cx(i, j)
                circuit.rz(2 * gamma * coeff, j)
                circuit.cx(i, j)

            # Mixer operator
            for i in range(self.n_qubits):
                circuit.rx(2 * beta, i)

            logger.debug(f"Added QAOA layer {p + 1}/{self.depth}")

        return circuit

    def get_expectation_values(self, params, cost_terms):
        """Get expectation values for all qubits.

        Computes the expectation values of the Z operator for each qubit
        using the current circuit parameters.

        Args:
            params: Circuit parameters
            cost_terms: List of (coefficient, (i, j)) tuples

        Returns:
            List[float]: Expectation values for each qubit
        """
        try:
            valid_terms = self._validate_cost_terms(cost_terms)
            if not valid_terms:
                raise ValueError("No valid cost terms for expectation value calculation")

            circuit = self._create_qaoa_circuit(params, valid_terms)

            # Create observables for each qubit
            observables = []
            for i in range(self.n_qubits):
                pauli_str = ['I'] * self.n_qubits
                pauli_str[i] = 'Z'
                obs = SparsePauliOp(''.join(pauli_str))
                observables.append(obs)

            # Calculate expectation values
            job = self.estimator.run([circuit] * self.n_qubits, observables)
            result = job.result()
            logger.debug(f"Computed {self.n_qubits} expectation values")

            return [float(val.real) for val in result.values]

        except Exception as e:
            logger.error("Error computing expectation values: %s", str(e))
            raise

    def optimize(self, cost_terms, steps=100):
        """Optimize the QAOA circuit parameters.

        Args:
            cost_terms: List of (coefficient, (i, j)) tuples
            steps: Number of optimization steps (default: 100)

        Returns:
            Tuple[np.ndarray, List[float]]: Optimal parameters and cost history
        """
        try:
            valid_terms = self._validate_cost_terms(cost_terms)
            if not valid_terms:
                raise ValueError("No valid cost terms for optimization")

            # Initialize parameters randomly
            params = np.random.uniform(-np.pi, np.pi, 2 * self.depth)
            costs = []

            def cost_function(p):
                """Quantum cost function for optimization."""
                measurements = self.get_expectation_values(p, valid_terms)
                cost = sum(coeff * measurements[i] * measurements[j] 
                          for coeff, (i, j) in valid_terms)
                return float(cost)

            # Simple optimization loop with adaptive learning rate
            alpha = 0.1  # Initial learning rate
            prev_cost = float('inf')

            for step in range(steps):
                current_cost = cost_function(params)
                costs.append(current_cost)

                # Adaptive learning rate
                if step > 0 and current_cost > prev_cost:
                    alpha *= 0.95  # Reduce learning rate if cost increases

                # Compute numerical gradient
                grad = np.zeros_like(params)
                eps = 1e-3
                for i in range(len(params)):
                    params_plus = params.copy()
                    params_plus[i] += eps
                    cost_plus = cost_function(params_plus)
                    grad[i] = (cost_plus - current_cost) / eps

                # Update parameters
                params -= alpha * grad
                prev_cost = current_cost

                if step % 10 == 0:
                    logger.info(f"Step {step}: Cost = {current_cost:.6f}, Learning rate = {alpha:.6f}")

            return params, costs

        except Exception as e:
            logger.error("Error during optimization: %s", str(e))
            raise