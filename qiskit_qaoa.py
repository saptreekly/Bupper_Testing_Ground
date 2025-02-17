import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.circuit import Parameter
from qiskit.primitives import BackendEstimator
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
            # Initialize parallel backends
            self.backend = Aer.get_backend('aer_simulator')
            self.backend.set_options(
                precision='double',
                max_parallel_threads=4,
                max_parallel_experiments=4
            )
            self.estimator = BackendEstimator(
                backend=self.backend,
                run_options={"shots": 1024},
                skip_transpilation=True  # We'll handle transpilation separately
            )
            logger.info("Initialized Qiskit backend with %d qubits", self.n_qubits)
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
        coeffs = [abs(coeff) for coeff, _ in cost_terms]
        max_coeff = max(coeffs)
        mean_coeff = sum(coeffs) / len(coeffs)
        threshold = mean_coeff * 0.01  # Adaptive threshold

        logger.debug(f"Validation thresholds - Max: {max_coeff:.2e}, Mean: {mean_coeff:.2e}")

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

        # Transpile circuit for optimization
        optimized_circuit = transpile(
            circuit,
            self.backend,
            optimization_level=3,
            seed_transpiler=42
        )

        return optimized_circuit

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

            # Run parallel estimation
            job = self.estimator.run([circuit] * self.n_qubits, observables)
            result = job.result()

            expectation_values = [float(val.real) for val in result.values]
            logger.debug(f"Computed {len(expectation_values)} expectation values")

            return expectation_values

        except Exception as e:
            logger.error("Error computing expectation values: %s", str(e))
            raise

    def optimize(self, cost_terms, steps=100):
        """Optimize the QAOA circuit parameters with adaptive optimization."""
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

            # Adaptive optimization parameters
            alpha = 0.1  # Initial learning rate
            min_alpha = 0.01
            patience = 5
            no_improvement = 0

            def cost_function(p):
                measurements = self.get_expectation_values(p, valid_terms)
                cost = sum(coeff * measurements[i] * measurements[j] 
                          for coeff, (i, j) in valid_terms)
                return float(cost)

            # Optimize with adaptive learning rate and momentum
            momentum = np.zeros_like(params)
            beta1 = 0.9  # Momentum coefficient

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

                # Compute gradient with parallel execution
                def compute_gradient_component(i):
                    eps = 1e-3
                    params_plus = params.copy()
                    params_plus[i] += eps
                    cost_plus = cost_function(params_plus)
                    return (cost_plus - current_cost) / eps

                with ThreadPoolExecutor(max_workers=4) as executor:
                    grad = list(executor.map(compute_gradient_component, range(len(params))))
                grad = np.array(grad)

                # Update with momentum
                momentum = beta1 * momentum + (1 - beta1) * grad
                params -= alpha * momentum

                if step % 10 == 0:
                    logger.info(f"Step {step}: Cost = {current_cost:.6f}, Learning rate = {alpha:.6f}")

            return best_params, costs

        except Exception as e:
            logger.error("Error during optimization: %s", str(e))
            raise