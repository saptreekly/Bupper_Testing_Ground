import numpy as np
import logging
from typing import List, Tuple, Optional
from scipy.optimize import minimize
from qiskit_qaoa import QiskitQAOA
from qaoa_core import QAOACircuit

logger = logging.getLogger(__name__)

class HybridOptimizer:
    def __init__(self, n_qubits: int, depth: int = 1, backend: str = 'qiskit', n_vehicles: int = 1):
        """Initialize hybrid optimizer with specified quantum backend."""
        try:
            self.n_qubits = n_qubits
            self.n_vehicles = n_vehicles
            self.depth = depth

            # Initialize parameters for optimization
            self.min_improvement = 1e-5
            self.convergence_window = max(3, min(10, n_qubits // 4))
            self.early_stopping_patience = max(5, min(20, n_qubits // 2))
            self.max_quantum_steps = max(10, min(50, n_qubits * 2))

            logger.info(f"Initializing {backend} optimizer with {n_qubits} qubits and depth {depth}")

            # Initialize backend with optimized settings
            self.backend = backend
            if backend == 'qiskit':
                try:
                    self.quantum_circuit = QiskitQAOA(n_qubits, depth)
                    logger.info("Successfully initialized Qiskit backend")
                except Exception as e:
                    logger.error(f"Failed to initialize Qiskit backend: {str(e)}")
                    raise
            else:
                try:
                    self.quantum_circuit = QAOACircuit(n_qubits, depth)
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

    def _classical_pre_optimization(self, cost_terms: List[Tuple]) -> np.ndarray:
        """Use classical optimization to find good initial parameters."""
        try:
            logger.info("Starting classical pre-optimization phase")

            def classical_cost(params):
                """Classical approximation of the quantum cost function."""
                try:
                    # Always use exactly 2 parameters
                    gamma, beta = params[0], params[1]
                    logger.debug(f"Classical cost evaluation with gamma={gamma:.4f}, beta={beta:.4f}")
                    cost = 0.0
                    for coeff, (i, j) in cost_terms:
                        # Use simpler approximation for initial parameters
                        zi = np.cos(gamma)
                        zj = np.cos(gamma)
                        cost += coeff * zi * zj
                    return float(cost)
                except Exception as e:
                    logger.error(f"Error in classical cost function: {str(e)}")
                    return float('inf')

            # Initialize with exactly 2 parameters
            initial_guess = np.array([np.pi/8, np.pi/4])  # [gamma, beta]
            logger.debug(f"Initial parameters: {initial_guess}")

            result = minimize(
                classical_cost,
                initial_guess,
                method='Nelder-Mead',
                options={'maxiter': 100, 'xatol': 1e-4}
            )

            if result.success:
                final_params = result.x[:2]  # Ensure exactly 2 parameters
                logger.info(f"Classical pre-optimization complete with params: {final_params}")
                return final_params
            else:
                logger.warning("Classical optimization failed, using initial guess")
                return initial_guess

        except Exception as e:
            logger.error(f"Error in classical pre-optimization: {str(e)}")
            return np.array([np.pi/8, np.pi/4])

    def optimize(self, cost_terms: List[Tuple], steps: int = 100, callback=None) -> Tuple[np.ndarray, List[float]]:
        """Run hybrid optimization process with improved convergence handling."""
        try:
            # Phase 1: Enhanced classical pre-optimization
            initial_params = self._classical_pre_optimization(cost_terms)
            logger.info(f"Starting optimization with initial parameters: {initial_params}")

            # Phase 2: Quantum optimization with adaptive steps
            quantum_steps = min(steps, self.max_quantum_steps)
            logger.info(f"Starting quantum optimization phase: {quantum_steps} steps")

            try:
                # Initialize history tracking
                cost_history = []
                best_cost = float('inf')
                best_params = None
                no_improvement_count = 0

                # Main optimization loop
                params = initial_params[:2]  # Ensure exactly 2 parameters
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

                        # Early stopping check
                        if no_improvement_count >= self.early_stopping_patience:
                            logger.info(f"Early stopping triggered after {step} steps")
                            break

                        # Update parameters using quantum gradient
                        new_params = self._quantum_gradient_step(params, cost_terms)
                        logger.debug(f"Updated parameters: {new_params}")
                        params = new_params[:2]  # Ensure exactly 2 parameters

                    except Exception as step_error:
                        logger.error(f"Error in optimization step {step}: {str(step_error)}")
                        continue

                final_params = best_params[:2] if best_params is not None else params[:2]
                logger.info(f"Optimization complete: best cost = {best_cost:.6f}, final params = {final_params}")

                return final_params, cost_history

            except Exception as quantum_error:
                logger.error(f"Quantum optimization failed: {str(quantum_error)}")
                raise

        except Exception as e:
            logger.error(f"Error in hybrid optimization: {str(e)}")
            raise

    def _quantum_gradient_step(self, params: np.ndarray, cost_terms: List[Tuple]) -> np.ndarray:
        """Compute quantum gradient and update parameters."""
        try:
            # Ensure we're only using 2 parameters (gamma, beta)
            if len(params) != 2:
                logger.warning(f"Received {len(params)} parameters, truncating to 2")
                params = params[:2]

            eps = 1e-4  # Finite difference step size
            grad = np.zeros_like(params)
            base_cost = self._evaluate_cost(params, cost_terms)

            # Compute gradient using finite differences
            for i in range(len(params)):
                params_plus = params.copy()
                params_plus[i] += eps
                cost_plus = self._evaluate_cost(params_plus, cost_terms)
                grad[i] = (cost_plus - base_cost) / eps

            # Normalize gradient
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1e-8:
                grad = grad / grad_norm

            # Update parameters with fixed learning rate
            learning_rate = 0.1
            new_params = params - learning_rate * grad

            # Ensure parameters stay within reasonable bounds
            new_params = np.clip(new_params, -2*np.pi, 2*np.pi)

            return new_params

        except Exception as e:
            logger.error(f"Error in quantum gradient step: {str(e)}")
            return params

    def _evaluate_cost(self, params: np.ndarray, cost_terms: List[Tuple]) -> float:
        """Evaluate cost function with error handling."""
        try:
            # Ensure we're only using 2 parameters (gamma, beta)
            if len(params) != 2:
                logger.warning(f"Received {len(params)} parameters, truncating to 2")
                params = params[:2]

            measurements = self.quantum_circuit.get_expectation_values(params, cost_terms)
            cost = sum(coeff * measurements[i] * measurements[j]
                      for coeff, (i, j) in cost_terms)
            return float(cost)

        except Exception as e:
            logger.error(f"Error evaluating cost function: {str(e)}")
            return float('inf')

    def get_expectation_values(self, params: np.ndarray, cost_terms: List[Tuple]) -> List[float]:
        """Get expectation values using the quantum circuit with improved error handling."""
        try:
            values = self.quantum_circuit.get_expectation_values(params, cost_terms)
            return values
        except Exception as e:
            logger.error(f"Error getting expectation values: {str(e)}")
            return [0.0] * self.n_qubits