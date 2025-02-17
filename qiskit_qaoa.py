import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.circuit import Parameter
from qiskit.primitives import BackendEstimatorV2 as BackendEstimator
from qiskit.quantum_info import SparsePauliOp
from qiskit.compiler import transpile
from concurrent.futures import ThreadPoolExecutor
import logging

logger = logging.getLogger(__name__)

class QiskitQAOA:
    """QAOA implementation using Qiskit with performance optimizations."""

    def __init__(self, n_qubits: int, depth: int = 1):
        """Initialize QAOA circuit with Qiskit backend."""
        self.n_qubits = min(n_qubits, 50)  # Hard limit at 50 qubits
        self.depth = depth
        try:
            # Initialize parallel backends with optimized settings
            self.backend = Aer.get_backend('aer_simulator')
            self.backend.set_options(
                precision='double',
                max_parallel_threads=4,
                max_parallel_experiments=4,
                shots=1024  # Configure shots at backend level
            )
            # Initialize BackendEstimatorV2 with the configured backend
            self.estimator = BackendEstimator(
                backend=self.backend
            )
            logger.info("Initialized Qiskit backend with %d qubits", self.n_qubits)
            logger.debug("Using simulator backend: %s", self.backend.name)
        except Exception as e:
            logger.error("Failed to initialize Qiskit backend: %s", str(e))
            raise

    def _validate_cost_terms(self, cost_terms):
        """Validate and normalize cost terms."""
        if not cost_terms:
            logger.warning("Empty cost terms provided")
            return []

        valid_terms = []
        seen_pairs = set()
        coeffs = [abs(coeff) for coeff, _ in cost_terms]
        max_coeff = max(coeffs)
        mean_coeff = sum(coeffs) / len(coeffs)
        threshold = 1e-14  # Even smaller threshold to keep most terms

        logger.debug(f"Validation thresholds - Max: {max_coeff:.2e}, Mean: {mean_coeff:.2e}, Threshold: {threshold:.2e}")
        logger.debug(f"Input cost terms: {cost_terms}")

        for coeff, (i, j) in cost_terms:
            if not (0 <= i < self.n_qubits and 0 <= j < self.n_qubits):
                logger.debug(f"Skipping invalid qubit indices: ({i}, {j})")
                continue

            pair = tuple(sorted([i, j]))
            if pair in seen_pairs:
                logger.debug(f"Skipping duplicate pair: {pair}")
                continue

            if abs(coeff) > threshold:
                norm_coeff = coeff / max_coeff if max_coeff > 0 else coeff
                valid_terms.append((norm_coeff, pair))
                seen_pairs.add(pair)
                logger.debug(f"Added cost term: {pair} with coefficient {norm_coeff:.6f}")

        logger.info(f"Validated {len(valid_terms)} cost terms from {len(cost_terms)} input terms")
        if not valid_terms:
            logger.warning("No valid cost terms found after validation!")
            raise ValueError("No valid cost terms for circuit construction")

        logger.debug(f"Valid terms: {valid_terms}")
        return valid_terms

    def _create_qaoa_circuit(self, params, cost_terms):
        """Create optimized QAOA circuit with given parameters."""
        valid_terms = self._validate_cost_terms(cost_terms)
        if not valid_terms:
            raise ValueError("No valid cost terms for circuit construction")

        circuit = QuantumCircuit(self.n_qubits)
        circuit.h(range(self.n_qubits))

        for p in range(self.depth):
            gamma = params[2 * p]
            beta = params[2 * p + 1]

            # Optimize cost operator implementation
            for coeff, (i, j) in valid_terms:
                if i != j:
                    circuit.cx(i, j)
                    circuit.rz(2 * gamma * coeff, j)
                    circuit.cx(i, j)

            # Optimize mixer operator
            circuit.rx(2 * beta, range(self.n_qubits))

        return circuit

    def get_expectation_values(self, params, cost_terms):
        """Get expectation values using parallel execution."""
        try:
            valid_terms = self._validate_cost_terms(cost_terms)
            if not valid_terms:
                raise ValueError("No valid cost terms for expectation value calculation")

            circuit = self._create_qaoa_circuit(params, valid_terms)
            observables = []

            # Prepare observables in parallel
            def create_observable(i):
                pauli_str = ['I'] * self.n_qubits
                pauli_str[i] = 'Z'
                return SparsePauliOp(''.join(pauli_str))

            with ThreadPoolExecutor(max_workers=4) as executor:
                observables = list(executor.map(create_observable, range(self.n_qubits)))

            # Package circuits and observables for BackendEstimatorV2
            circuit_observables = [(circuit, obs) for obs in observables]
            job = self.estimator.run(circuit_observables)
            result = job.result()

            # Extract values from result using data() method
            logger.debug("Result data structure: %s", dir(result))
            data = result.data()
            expectation_values = []

            # Process each measurement result
            for meas_data in data:
                if hasattr(meas_data, 'expval'):
                    # Direct expectation value
                    exp_val = float(meas_data.expval.real)
                else:
                    # Calculate from counts
                    counts = meas_data.get('counts', {})
                    total_shots = sum(counts.values())
                    exp_val = sum((-1 if bin(state).count('1') % 2 else 1) * count / total_shots 
                                for state, count in counts.items())
                expectation_values.append(exp_val)

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

            # Improved parameter initialization
            params = np.zeros(2 * self.depth)
            for i in range(self.depth):
                params[2*i] = np.random.uniform(-np.pi/8, np.pi/8)  # gamma
                params[2*i+1] = np.pi/4 + np.random.uniform(-np.pi/8, np.pi/8)  # beta

            costs = []
            best_params = None
            best_cost = float('inf')

            def cost_function(p):
                measurements = self.get_expectation_values(p, valid_terms)
                cost = sum(coeff * measurements[i] * measurements[j] 
                          for coeff, (i, j) in valid_terms)
                return float(cost)

            # Optimize with momentum
            alpha = 0.1  # Initial learning rate
            momentum = np.zeros_like(params)
            beta1 = 0.9  # Momentum coefficient

            for step in range(steps):
                current_cost = cost_function(params)
                costs.append(current_cost)

                if current_cost < best_cost:
                    best_cost = current_cost
                    best_params = params.copy()

                # Compute gradient
                eps = 1e-3
                grad = np.zeros_like(params)
                for i in range(len(params)):
                    params_plus = params.copy()
                    params_plus[i] += eps
                    cost_plus = cost_function(params_plus)
                    grad[i] = (cost_plus - current_cost) / eps

                # Update with momentum
                momentum = beta1 * momentum + (1 - beta1) * grad
                params -= alpha * momentum

                if step % 10 == 0:
                    logger.info(f"Step {step}: Cost = {current_cost:.6f}")

            return best_params, costs

        except Exception as e:
            logger.error("Error during optimization: %s", str(e))
            raise