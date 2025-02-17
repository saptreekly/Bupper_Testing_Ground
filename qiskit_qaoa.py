import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.circuit import Parameter
from qiskit.primitives import StatevectorEstimator
from qiskit.result import Result
from qiskit.quantum_info import SparsePauliOp
from qiskit.compiler import transpile
from concurrent.futures import ThreadPoolExecutor
import logging

logger = logging.getLogger(__name__)

class QiskitQAOA:
    def __init__(self, n_qubits: int, depth: int = 1):
        """Initialize QAOA circuit with Qiskit backend."""
        self.n_qubits = n_qubits
        self.depth = depth
        try:
            # Configure backend with optimized settings for larger problems
            self.backend = Aer.get_backend('aer_simulator_statevector')
            self.backend.set_options(
                precision='double',
                max_parallel_threads=8,  # Increased parallelism
                max_parallel_experiments=8,
                max_parallel_shots=1024,
                shots=2048  # Increased shots for better statistics
            )

            # Use StatevectorEstimator without options (they are set on the backend)
            self.estimator = StatevectorEstimator()
            logger.info(f"Initialized Qiskit backend with {n_qubits} qubits and depth {depth}")
            logger.debug(f"Backend options: {self.backend.options}")
        except Exception as e:
            logger.error("Failed to initialize Qiskit backend: %s", str(e))
            raise

    def _validate_cost_terms(self, cost_terms):
        """Validate and normalize cost terms with improved handling of minimal cases."""
        if not cost_terms:
            logger.warning("Empty cost terms provided")
            return []

        valid_terms = []
        seen_pairs = set()

        # Use absolute values for coefficient comparison
        coeffs = [abs(coeff) for coeff, _ in cost_terms]
        if not coeffs:
            logger.warning("No coefficients found in cost terms")
            return []

        max_coeff = max(coeffs)
        min_coeff = min(coeffs)
        mean_coeff = sum(coeffs) / len(coeffs)

        # Log detailed statistics
        logger.debug(f"Cost terms statistics - max: {max_coeff:.2e}, min: {min_coeff:.2e}, mean: {mean_coeff:.2e}")
        logger.debug(f"Number of terms: {len(cost_terms)}")

        # Adaptive threshold based on problem size
        threshold = max_coeff * (1e-8 if len(cost_terms) > 10 else 1e-6)
        logger.debug(f"Using threshold: {threshold:.2e}")

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

        # Always ensure at least one valid term
        if not valid_terms and self.n_qubits > 1:
            valid_terms.append((1.0, (0, 1)))
            logger.info("Added minimal interaction term")

        logger.info(f"Validated {len(valid_terms)} cost terms")
        return valid_terms

    def _create_qaoa_circuit(self, params, cost_terms):
        """Create optimized QAOA circuit with given parameters."""
        try:
            valid_terms = self._validate_cost_terms(cost_terms)
            if not valid_terms:
                logger.warning("No valid cost terms available, using minimal circuit")
                circuit = QuantumCircuit(self.n_qubits)
                circuit.h(range(self.n_qubits))
                circuit.rz(params[0], 0)
                circuit.rx(params[1], range(self.n_qubits))
                return circuit

            circuit = QuantumCircuit(self.n_qubits)
            circuit.h(range(self.n_qubits))

            for p in range(self.depth):
                gamma = params[2 * p]
                beta = params[2 * p + 1]

                # Problem Hamiltonian with batched operations
                for coeff, (i, j) in valid_terms:
                    if i != j:
                        circuit.cx(i, j)
                        circuit.rz(2 * gamma * coeff, j)
                        circuit.cx(i, j)
                    else:
                        circuit.rz(gamma * coeff, i)

                # Batched mixing Hamiltonian
                circuit.rx(2 * beta, range(self.n_qubits))

            logger.debug(f"Created QAOA circuit with {circuit.size()} gates")
            return circuit

        except Exception as e:
            logger.error(f"Error creating QAOA circuit: {str(e)}")
            raise

    def get_expectation_values(self, params, cost_terms):
        """Get expectation values using optimized parallel execution."""
        try:
            valid_terms = self._validate_cost_terms(cost_terms)
            circuit = self._create_qaoa_circuit(params, valid_terms)

            # Create Z observables efficiently
            observables = [SparsePauliOp(''.join(['I'] * i + ['Z'] + ['I'] * (self.n_qubits - i - 1)))
                         for i in range(self.n_qubits)]

            # Execute measurement job
            job = self.estimator.run([circuit] * len(observables), observables)
            result = job.result()

            # Extract expectation values with error handling
            try:
                exp_vals = [float(val) for val in result.values]
                logger.debug(f"Extracted {len(exp_vals)} expectation values")
                return exp_vals
            except Exception as e:
                logger.error(f"Error extracting expectation values: {str(e)}")
                logger.warning("Using fallback expectation values")
                return [1.0] * self.n_qubits

        except Exception as e:
            logger.error(f"Error computing expectation values: {str(e)}")
            raise

    def optimize(self, cost_terms, steps=100):
        """Optimize the QAOA circuit parameters with improved convergence."""
        try:
            # Validate and normalize cost terms
            valid_terms = self._validate_cost_terms(cost_terms)
            if not valid_terms:
                valid_terms = [(1.0, (0, min(1, self.n_qubits-1)))]

            # Initialize parameters with improved strategy
            params = []
            for _ in range(self.depth):
                params.extend([
                    np.random.uniform(-np.pi/8, np.pi/8),  # gamma
                    np.pi/4 + np.random.uniform(-np.pi/8, np.pi/8)  # beta
                ])
            params = np.array(params)
            logger.info(f"Initial parameters: {params}")

            costs = []
            best_params = None
            best_cost = float('inf')
            no_improvement_count = 0

            def cost_function(p):
                """Compute cost with error handling."""
                try:
                    measurements = self.get_expectation_values(p, valid_terms)
                    cost = sum(coeff * measurements[i] * measurements[j]
                             for coeff, (i, j) in valid_terms)
                    return float(cost)
                except Exception as e:
                    logger.error(f"Error in cost function: {str(e)}")
                    return float('inf')

            # Optimize with adaptive learning rate
            alpha = 0.1
            momentum = np.zeros_like(params)
            beta1 = 0.9

            for step in range(steps):
                try:
                    current_cost = cost_function(params)
                    costs.append(current_cost)

                    if current_cost < best_cost:
                        improvement = (best_cost - current_cost) / abs(best_cost) if best_cost != float('inf') else 1.0
                        best_cost = current_cost
                        best_params = params.copy()
                        no_improvement_count = 0
                        logger.info(f"Step {step}: New best cost = {current_cost:.6f} (improved by {improvement:.1%})")
                    else:
                        no_improvement_count += 1

                    # Early stopping check
                    if no_improvement_count >= 20:
                        logger.info(f"Early stopping at step {step}")
                        break

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

                except Exception as e:
                    logger.error(f"Error in optimization step {step}: {str(e)}")
                    continue

            return best_params if best_params is not None else params, costs

        except Exception as e:
            logger.error(f"Error during optimization: {str(e)}")
            raise