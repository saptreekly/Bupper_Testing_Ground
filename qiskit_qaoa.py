import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import Aer
from qiskit.circuit import Parameter
from qiskit.primitives import BackendEstimator
from qiskit.quantum_info import SparsePauliOp
import logging

logger = logging.getLogger(__name__)

class QiskitQAOA:
    """QAOA implementation using Qiskit."""

    def __init__(self, n_qubits: int, depth: int = 1):
        """Initialize QAOA circuit with Qiskit backend."""
        self.n_qubits = min(n_qubits, 25)  # Hard limit at 25 qubits
        self.depth = depth
        try:
            self.backend = Aer.get_backend('aer_simulator')
            self.estimator = BackendEstimator(backend=self.backend)
            logger.info("Initialized Qiskit backend with %d qubits", self.n_qubits)
            # Use name attribute instead of backend_name
            logger.debug("Using simulator backend: %s", self.backend.name)
        except Exception as e:
            logger.error("Failed to initialize Qiskit backend: %s", str(e))
            raise

    def _validate_cost_terms(self, cost_terms):
        """Validate and normalize cost terms with improved filtering."""
        if not cost_terms:
            logger.warning("Empty cost terms provided")
            return []

        valid_terms = []
        seen_pairs = set()
        max_coeff = max(abs(coeff) for coeff, _ in cost_terms)

        if max_coeff < 1e-10:
            logger.warning("All coefficients are nearly zero")
            return []

        threshold = max_coeff * 1e-4  # Increased threshold for numerical stability
        logger.debug(f"Validation threshold: {threshold:.2e}")

        for coeff, (i, j) in cost_terms:
            if not (0 <= i < self.n_qubits and 0 <= j < self.n_qubits):
                logger.debug(f"Skipping invalid qubit indices: ({i}, {j})")
                continue

            pair = tuple(sorted([i, j]))
            if pair in seen_pairs:
                logger.debug(f"Skipping duplicate pair: {pair}")
                continue

            if abs(coeff) > threshold:
                norm_coeff = coeff / max_coeff
                valid_terms.append((norm_coeff, pair))
                seen_pairs.add(pair)
                logger.debug(f"Added cost term: {pair} with coefficient {norm_coeff:.6f}")

        logger.info(f"Validated {len(valid_terms)} cost terms from {len(cost_terms)} input terms")
        return valid_terms

    def _create_qaoa_circuit(self, params, cost_terms):
        """Create QAOA circuit with given parameters."""
        valid_terms = self._validate_cost_terms(cost_terms)
        if not valid_terms:
            raise ValueError("No valid cost terms for circuit construction")

        circuit = QuantumCircuit(self.n_qubits)
        circuit.h(range(self.n_qubits))
        logger.debug("Initialized circuit in uniform superposition state")

        for p in range(self.depth):
            gamma = params[2 * p]
            beta = params[2 * p + 1]

            # Cost operator with improved implementation
            for coeff, (i, j) in valid_terms:
                if i != j:  # Skip self-interactions
                    circuit.cx(i, j)
                    circuit.rz(2 * gamma * coeff, j)
                    circuit.cx(i, j)

            # Mixer operator
            for i in range(self.n_qubits):
                circuit.rx(2 * beta, i)

            logger.debug(f"Completed QAOA layer {p + 1}/{self.depth}")

        return circuit

    def get_expectation_values(self, params, cost_terms):
        """Get expectation values for all qubits."""
        try:
            valid_terms = self._validate_cost_terms(cost_terms)
            if not valid_terms:
                raise ValueError("No valid cost terms for expectation value calculation")

            circuit = self._create_qaoa_circuit(params, valid_terms)
            observables = []

            for i in range(self.n_qubits):
                pauli_str = ['I'] * self.n_qubits
                pauli_str[i] = 'Z'
                obs = SparsePauliOp(''.join(pauli_str))
                observables.append(obs)

            job = self.estimator.run([circuit] * self.n_qubits, observables)
            result = job.result()

            expectation_values = [float(val.real) for val in result.values]
            logger.debug(f"Computed {len(expectation_values)} expectation values")

            return expectation_values

        except Exception as e:
            logger.error("Error computing expectation values: %s", str(e))
            raise

    def optimize(self, cost_terms, steps=100):
        """Optimize the QAOA circuit parameters."""
        try:
            valid_terms = self._validate_cost_terms(cost_terms)
            if not valid_terms:
                raise ValueError("No valid cost terms for optimization")

            # Initialize parameters with improved scaling
            params = np.random.uniform(-np.pi/4, np.pi/4, 2 * self.depth)
            costs = []
            best_params = None
            best_cost = float('inf')

            def cost_function(p):
                measurements = self.get_expectation_values(p, valid_terms)
                cost = sum(coeff * measurements[i] * measurements[j] 
                          for coeff, (i, j) in valid_terms)
                return float(cost)

            # Adaptive optimization
            alpha = 0.1  # Initial learning rate
            min_alpha = 0.01
            patience = 5
            no_improvement = 0

            for step in range(steps):
                current_cost = cost_function(params)
                costs.append(current_cost)

                # Update best solution
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_params = params.copy()
                    no_improvement = 0
                else:
                    no_improvement += 1

                # Adaptive learning rate
                if no_improvement >= patience:
                    alpha = max(alpha * 0.8, min_alpha)
                    no_improvement = 0
                    logger.debug(f"Reduced learning rate to {alpha:.6f}")

                # Compute gradient
                grad = np.zeros_like(params)
                eps = 1e-3
                for i in range(len(params)):
                    params_plus = params.copy()
                    params_plus[i] += eps
                    cost_plus = cost_function(params_plus)
                    grad[i] = (cost_plus - current_cost) / eps

                # Update parameters
                params -= alpha * grad

                if step % 10 == 0:
                    logger.info(f"Step {step}: Cost = {current_cost:.6f}, Learning rate = {alpha:.6f}")

            return best_params, costs

        except Exception as e:
            logger.error("Error during optimization: %s", str(e))
            raise