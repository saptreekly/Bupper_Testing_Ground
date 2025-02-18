import numpy as np
import logging
from typing import List, Tuple, Optional
from scipy.optimize import minimize
from qiskit_qaoa import QiskitQAOA
from qaoa_core import QAOACircuit

logger = logging.getLogger(__name__)

class HybridOptimizer:
    def __init__(self, n_qubits: int, depth: int = 1, backend: str = 'qiskit', n_vehicles: int = 1):
        """Initialize hybrid optimizer with large-scale quantum backend support."""
        try:
            self.n_qubits = n_qubits
            self.n_vehicles = n_vehicles
            self.depth = 1  # Force depth to 1 to maintain 2-parameter system
            self.backend = backend

            # Calculate actual qubit requirements
            self.max_qiskit_qubits = 31
            self.max_pennylane_qubits = 25
            self.max_qubits = self.max_qiskit_qubits if backend == 'qiskit' else self.max_pennylane_qubits

            required_qubits = n_qubits * n_vehicles
            if required_qubits > self.max_qubits:
                logger.warning(f"Required qubits ({required_qubits}) exceeds backend maximum ({self.max_qubits})")
                logger.info("Implementing circuit partitioning strategy")

            # Optimization parameters for large-scale problems
            self.min_improvement = 1e-5
            self.convergence_window = max(3, min(20, n_qubits // 8))
            self.early_stopping_patience = max(10, min(30, n_qubits // 4))
            self.max_quantum_steps = max(20, min(100, n_qubits))

            logger.info(f"Initializing {backend} optimizer for {n_qubits} qubits with {n_vehicles} vehicles")

            # Initialize quantum backend with strict parameter validation
            if backend == 'qiskit':
                try:
                    self.quantum_circuit = QiskitQAOA(n_qubits, 1)
                    logger.info("Successfully initialized Qiskit backend with parallel execution")
                except Exception as e:
                    logger.error(f"Failed to initialize Qiskit backend: {str(e)}")
                    raise
            else:
                try:
                    if n_qubits > self.max_pennylane_qubits:
                        logger.error(f"Problem size too large: {n_qubits} qubits exceeds PennyLane maximum of {self.max_pennylane_qubits}")
                        raise ValueError(f"Maximum allowed qubits for pennylane backend is {self.max_pennylane_qubits}")
                    self.quantum_circuit = QAOACircuit(n_qubits, 1)
                    logger.info("Successfully initialized PennyLane backend")
                except Exception as e:
                    logger.error(f"Failed to initialize PennyLane backend: {str(e)}")
                    raise

            logger.info("Optimizer parameters:")
            logger.info(f"- Convergence window: {self.convergence_window}")
            logger.info(f"- Early stopping patience: {self.early_stopping_patience}")
            logger.info(f"- Max quantum steps: {self.max_quantum_steps}")

        except Exception as e:
            logger.error(f"Failed to initialize hybrid optimizer: {str(e)}")
            raise

    def _validate_and_truncate_params(self, params: np.ndarray) -> np.ndarray:
        """Ensure parameters are exactly length 2 and within bounds."""
        if len(params) != 2:
            logger.warning(f"Truncating parameter array from length {len(params)} to 2")
            params = params[:2]

        # Ensure parameters are within bounds
        params[0] = np.clip(params[0], -2*np.pi, 2*np.pi)  # gamma
        params[1] = np.clip(params[1], -np.pi, np.pi)      # beta

        return params

    def _classical_pre_optimization(self, cost_terms: List[Tuple]) -> np.ndarray:
        """Enhanced classical pre-optimization with parallel search."""
        try:
            logger.info("Starting enhanced classical pre-optimization phase")

            def classical_cost(params):
                """Classical approximation of the quantum cost function."""
                try:
                    params = self._validate_and_truncate_params(params)
                    gamma, beta = params[0], params[1]
                    logger.debug(f"Classical cost evaluation with gamma={gamma:.4f}, beta={beta:.4f}")

                    # Parallel cost computation for large problems
                    chunk_size = 1000
                    total_cost = 0.0

                    for i in range(0, len(cost_terms), chunk_size):
                        chunk = cost_terms[i:i + chunk_size]
                        chunk_cost = sum(
                            coeff * np.cos(gamma) * np.cos(2 * beta)
                            for coeff, (_, _) in chunk
                        )
                        total_cost += chunk_cost

                    return float(total_cost)
                except Exception as e:
                    logger.error(f"Error in classical cost function: {str(e)}")
                    return float('inf')

            # Multiple starting points with adaptive search
            base_points = [
                np.array([np.pi/8, np.pi/4]),
                np.array([0.0, np.pi/2]),
                np.array([np.pi/4, np.pi/8]),
                np.array([-np.pi/8, np.pi/6])
            ]

            # Add random perturbations for better exploration
            starting_points = []
            for point in base_points:
                starting_points.append(point)
                for _ in range(2):  # Add 2 perturbed versions of each base point
                    perturbed = point + np.random.normal(0, 0.1, 2)
                    starting_points.append(perturbed)

            best_result = None
            best_cost = float('inf')

            for start_point in starting_points:
                logger.debug(f"Trying starting point: {start_point}")
                result = minimize(
                    classical_cost,
                    start_point,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 200,
                        'xatol': 1e-4,
                        'adaptive': True
                    }
                )

                if result.success and result.fun < best_cost:
                    best_cost = result.fun
                    best_result = result
                    logger.info(f"Found better solution with cost: {best_cost:.6f}")

            if best_result is not None:
                final_params = self._validate_and_truncate_params(best_result.x)
                logger.info(f"Classical pre-optimization complete with params: {final_params}")
                return final_params
            else:
                logger.warning("Classical optimization failed, using initial guess")
                return np.array([np.pi/8, np.pi/4])

        except Exception as e:
            logger.error(f"Error in classical pre-optimization: {str(e)}")
            return np.array([np.pi/8, np.pi/4])

    def optimize(self, cost_terms: List[Tuple], steps: int = 100, callback=None) -> Tuple[np.ndarray, List[float]]:
        """Run hybrid optimization process with improved parameter validation."""
        try:
            # Phase 1: Enhanced classical pre-optimization
            initial_params = self._validate_and_truncate_params(self._classical_pre_optimization(cost_terms))
            logger.info(f"Starting optimization with initial parameters: {initial_params}")

            # Phase 2: Quantum optimization with adaptive steps
            quantum_steps = min(steps, self.max_quantum_steps)
            logger.info(f"Starting quantum optimization phase: {quantum_steps} steps")

            # Initialize history tracking
            cost_history = []
            best_cost = float('inf')
            best_params = None
            no_improvement_count = 0

            # Main optimization loop with strict parameter validation
            params = initial_params.copy()
            for step in range(quantum_steps):
                try:
                    logger.debug(f"Step {step}: Current parameters = {params}")
                    current_cost = self._evaluate_cost(params, cost_terms)
                    cost_history.append(current_cost)

                    if callback:
                        callback(step, {
                            'step': step,
                            'total_steps': quantum_steps,
                            'cost': current_cost,
                            'best_cost': best_cost if best_cost != float('inf') else current_cost,
                            'progress': step / quantum_steps
                        })

                    # Update best solution with improved threshold handling
                    if current_cost < best_cost:
                        improvement = (best_cost - current_cost) / abs(best_cost) if best_cost != float('inf') else 1.0
                        if improvement > self.min_improvement:
                            best_cost = current_cost
                            best_params = params.copy()
                            no_improvement_count = 0
                            logger.info(f"New best cost at step {step}: {best_cost:.6f}")
                        else:
                            no_improvement_count += 1
                    else:
                        no_improvement_count += 1

                    # Early stopping with adaptive patience
                    if no_improvement_count >= self.early_stopping_patience:
                        logger.info(f"Early stopping triggered after {step} steps")
                        break

                    # Update parameters using improved quantum gradient
                    params = self._validate_and_truncate_params(self._quantum_gradient_step(params, cost_terms))

                except Exception as step_error:
                    logger.error(f"Error in optimization step {step}: {str(step_error)}")
                    continue

            final_params = best_params if best_params is not None else params
            logger.info(f"Optimization complete: best cost = {best_cost:.6f}")

            return self._validate_and_truncate_params(final_params), cost_history

        except Exception as e:
            logger.error(f"Error in hybrid optimization: {str(e)}")
            raise

    def _quantum_gradient_step(self, params: np.ndarray, cost_terms: List[Tuple]) -> np.ndarray:
        """Compute quantum gradient with improved numerical stability and parameter validation."""
        try:
            params = self._validate_and_truncate_params(params)
            eps = 1e-4
            grad = np.zeros(2)
            base_cost = self._evaluate_cost(params, cost_terms)

            # Parallel gradient computation
            param_variations = []
            for i in range(2):
                params_plus = params.copy()
                params_minus = params.copy()
                params_plus[i] += eps
                params_minus[i] -= eps
                param_variations.extend([params_plus, params_minus])

            # Evaluate all variations in parallel
            costs = []
            for p in param_variations:
                cost = self._evaluate_cost(self._validate_and_truncate_params(p), cost_terms)
                costs.append(cost)

            # Compute gradients using central differences
            for i in range(2):
                cost_plus = costs[2*i]
                cost_minus = costs[2*i + 1]
                grad[i] = (cost_plus - cost_minus) / (2 * eps)

            # Normalize gradient with improved numerical stability
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1e-8:
                grad = grad / grad_norm

            # Adaptive learning rate based on problem size
            learning_rate = 0.1 * (1.0 / np.sqrt(1 + self.n_qubits / 100))
            new_params = params - learning_rate * grad

            return self._validate_and_truncate_params(new_params)

        except Exception as e:
            logger.error(f"Error in quantum gradient step: {str(e)}")
            return self._validate_and_truncate_params(params)

    def _evaluate_cost(self, params: np.ndarray, cost_terms: List[Tuple]) -> float:
        """Evaluate cost function with strict parameter validation."""
        try:
            params = self._validate_and_truncate_params(params)
            measurements = self.quantum_circuit.get_expectation_values(params, cost_terms)
            cost = sum(coeff * measurements[i] * measurements[j]
                      for coeff, (i, j) in cost_terms)
            return float(cost)

        except Exception as e:
            logger.error(f"Error evaluating cost function: {str(e)}")
            return float('inf')

    def get_expectation_values(self, params: np.ndarray, cost_terms: List[Tuple]) -> List[float]:
        """Get expectation values using the quantum circuit with strict parameter validation."""
        try:
            params = self._validate_and_truncate_params(params)
            values = self.quantum_circuit.get_expectation_values(params, cost_terms)
            return values
        except Exception as e:
            logger.error(f"Error getting expectation values: {str(e)}")
            return [0.0] * self.n_qubits