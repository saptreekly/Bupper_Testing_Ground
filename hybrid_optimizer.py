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

            # Calculate optimal circuit depth based on problem size
            base_depth = max(1, min(3, n_qubits // 8))  # Scale depth with problem size
            vehicle_factor = max(1, n_vehicles // 2)  # Scale depth with vehicle count
            noise_factor = 0.4 if backend == 'qiskit' else 0.6  # PennyLane allows deeper circuits

            self.depth = max(1, min(
                depth,
                int(base_depth * vehicle_factor * noise_factor)
            ))

            logger.info(f"Enhanced adaptive circuit depth calculation:")
            logger.info(f"- Problem size: {n_qubits} qubits")
            logger.info(f"- Base depth: {base_depth}")
            logger.info(f"- Depth scale: {vehicle_factor}")
            logger.info(f"- Noise factor: {noise_factor}")
            logger.info(f"- Final depth: {self.depth}")

            # Create size-dependent noise model
            self.noise_scale = min(0.4, max(0.1, n_qubits / 100))
            logger.info(f"Created enhanced noise model with size factor {self.noise_scale:.3f}")

            # Initialize backend with optimized settings
            self.backend = backend
            if backend == 'qiskit':
                try:
                    self.quantum_circuit = QiskitQAOA(
                        n_qubits,
                        self.depth,
                        noise_scale=self.noise_scale
                    )
                    logger.info("Successfully created custom noise model")
                except Exception as e:
                    logger.error(f"Failed to initialize Qiskit backend: {str(e)}")
                    raise
            else:
                try:
                    self.quantum_circuit = QAOACircuit(
                        n_qubits,
                        self.depth,
                        noise_scale=self.noise_scale
                    )
                except Exception as e:
                    logger.error(f"Failed to initialize PennyLane backend: {str(e)}")
                    raise

            # Optimization hyperparameters
            self.min_improvement = 1e-5  # Reduced threshold for more iterations
            self.convergence_window = max(3, min(10, n_qubits // 4))
            self.early_stopping_patience = max(5, min(20, n_qubits // 2))
            self.max_quantum_steps = max(10, min(50, n_qubits * 2))

            logger.info(f"Initialized {backend} optimizer with {n_qubits} qubits")
            logger.info(f"Optimization parameters:")
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
                    cost = 0.0
                    for coeff, (i, j) in cost_terms:
                        # Use simpler approximation for initial parameters
                        zi = np.cos(params[0])
                        zj = np.cos(params[0])
                        cost += coeff * zi * zj
                    return float(cost)
                except Exception as e:
                    logger.error(f"Error in classical cost function: {str(e)}")
                    return float('inf')

            # Multiple starts with improved parameter ranges
            best_cost = float('inf')
            best_params = None
            n_starts = 3  # Reduced number of starts for efficiency

            for start in range(n_starts):
                try:
                    # Initialize with smaller parameter range - exactly 2 parameters regardless of depth
                    initial_guess = np.random.uniform(-np.pi/4, np.pi/4, 2)  # Always 2 parameters
                    logger.debug(f"Start {start + 1}/{n_starts}: Initial parameters: {initial_guess}")

                    result = minimize(
                        classical_cost,
                        initial_guess,
                        method='Nelder-Mead',  # Changed to more robust optimizer
                        options={'maxiter': 100, 'xatol': 1e-4}
                    )

                    if result.fun < best_cost:
                        best_cost = result.fun
                        best_params = result.x
                        logger.debug(f"New best classical solution: cost = {best_cost:.6f}")
                except Exception as e:
                    logger.warning(f"Failed optimization attempt {start + 1}: {str(e)}")
                    continue

            if best_params is None:
                # Fallback to simple initialization if optimization fails
                logger.warning("Classical optimization failed, using fallback initialization")
                best_params = np.array([np.pi/8, np.pi/4])  # Always 2 parameters

            logger.info(f"Classical pre-optimization complete: cost = {best_cost:.6f}")
            return best_params

        except Exception as e:
            logger.error(f"Error in classical pre-optimization: {str(e)}")
            raise

    def _calculate_capacity_penalty(self, params: np.ndarray) -> float:
        """Calculate penalty for violating vehicle capacity constraints."""
        try:
            if self.n_vehicles <= 1:
                return 0.0

            penalty = 0.0
            # Use binary encoding for vehicle assignment
            n_encoding_qubits = int(np.ceil(np.log2(self.n_vehicles + 1)))
            vehicle_loads = np.zeros(self.n_vehicles)

            # Calculate loads using optimized parameter mapping
            param_per_vehicle = 2 * self.depth // self.n_vehicles
            for v in range(self.n_vehicles):
                start_idx = v * param_per_vehicle
                end_idx = (v + 1) * param_per_vehicle
                vehicle_params = params[start_idx:end_idx]
                if len(vehicle_params) > 0:
                    load_estimate = np.mean(np.cos(vehicle_params))
                    vehicle_loads[v] = load_estimate

            # Calculate penalties with improved scaling
            mean_load = np.mean(vehicle_loads)
            load_variance = np.var(vehicle_loads)

            # Scale penalties based on problem size
            variance_weight = 5.0 / self.n_vehicles
            capacity_weight = 50.0 / self.n_vehicles

            # Add scaled penalties
            penalty += load_variance * variance_weight
            max_capacity = 1.0 / self.n_vehicles
            capacity_violations = np.maximum(0, vehicle_loads - max_capacity)
            penalty += np.sum(capacity_violations) * capacity_weight

            return float(penalty)
        except Exception as e:
            logger.error(f"Error calculating capacity penalty: {str(e)}")
            return 0.0

    def optimize(self, cost_terms: List[Tuple], steps: int = 100, callback=None) -> Tuple[np.ndarray, List[float]]:
        """Run hybrid optimization process with improved convergence handling."""
        try:
            # Phase 1: Enhanced classical pre-optimization
            initial_params = None
            max_retries = 3

            for attempt in range(max_retries):
                try:
                    initial_params = self._classical_pre_optimization(cost_terms)
                    break
                except Exception as e:
                    logger.warning(f"Pre-optimization attempt {attempt + 1} failed: {str(e)}")
                    if attempt == max_retries - 1:
                        logger.error("All pre-optimization attempts failed")
                        raise
                    continue

            logger.info("Using classically optimized initial parameters")
            logger.debug(f"Initial parameters: {initial_params}")

            # Phase 2: Quantum optimization with adaptive steps
            quantum_steps = min(steps, self.max_quantum_steps)
            logger.info(f"Starting quantum optimization phase: {quantum_steps} steps")

            try:
                # Initialize history tracking
                cost_history = []
                param_history = []
                best_cost = float('inf')
                best_params = None
                no_improvement_count = 0

                # Main optimization loop with batch processing
                batch_size = min(5, quantum_steps // 2)
                for step in range(0, quantum_steps, batch_size):
                    batch_costs = []
                    batch_params = []

                    # Process multiple parameter sets in parallel
                    for i in range(batch_size):
                        if step + i >= quantum_steps:
                            break

                        current_params = initial_params if step + i == 0 else param_history[-1]
                        current_cost = self._evaluate_cost(current_params, cost_terms)

                        batch_costs.append(current_cost)
                        batch_params.append(current_params.copy())

                        if callback:
                            progress = (step + i) / quantum_steps
                            callback(step + i, {
                                'step': step + i,
                                'total_steps': quantum_steps,
                                'cost': current_cost,
                                'best_cost': best_cost if best_cost != float('inf') else current_cost,
                                'progress': progress
                            })

                    # Update histories
                    cost_history.extend(batch_costs)
                    param_history.extend(batch_params)

                    # Find best result in batch
                    batch_best_idx = np.argmin(batch_costs)
                    batch_best_cost = batch_costs[batch_best_idx]

                    if batch_best_cost < best_cost:
                        improvement = (best_cost - batch_best_cost) / abs(best_cost) if best_cost != float('inf') else 1.0
                        if improvement > self.min_improvement:
                            best_cost = batch_best_cost
                            best_params = batch_params[batch_best_idx].copy()
                            no_improvement_count = 0
                            logger.info(f"New best cost at step {step}: {best_cost:.6f}")
                        else:
                            no_improvement_count += 1
                    else:
                        no_improvement_count += 1

                    # Early stopping check
                    if no_improvement_count >= self.early_stopping_patience:
                        logger.info(f"Early stopping triggered after {step + batch_size} steps")
                        break

                    # Convergence check
                    if len(cost_history) >= self.convergence_window:
                        recent_costs = cost_history[-self.convergence_window:]
                        if np.std(recent_costs) < self.min_improvement:
                            logger.info(f"Convergence achieved after {step + batch_size} steps")
                            break

                    # Update parameters using quantum gradient
                    new_params = self._quantum_gradient_step(
                        batch_params[batch_best_idx],
                        cost_terms
                    )
                    initial_params = new_params

                final_params = best_params if best_params is not None else initial_params
                logger.info(f"Quantum optimization phase complete: best cost = {best_cost:.6f}")

                # Phase 3: Classical refinement with adaptive stopping
                result = minimize(
                    lambda p: self._evaluate_cost(p, cost_terms),
                    final_params,
                    method='COBYLA',
                    options={'maxiter': min(20, steps - quantum_steps)}
                )

                final_params = result.x
                final_cost = result.fun
                improvement = (best_cost - final_cost) / abs(best_cost) if best_cost != float('inf') else 0
                logger.info(f"Classical refinement improved cost by {improvement:.1%}")

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

            # Update parameters with adaptive learning rate
            learning_rate = 0.1  # Fixed learning rate for 2 parameters
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
            measurements = self.quantum_circuit.get_expectation_values(params, cost_terms)
            cost = sum(coeff * measurements[i] * measurements[j]
                      for coeff, (i, j) in cost_terms)

            # Add vehicle capacity constraints for multi-vehicle problems
            if self.n_vehicles > 1:
                capacity_penalty = self._calculate_capacity_penalty(params)
                cost += capacity_penalty

            return float(cost)

        except Exception as e:
            logger.error(f"Error evaluating cost function: {str(e)}")
            return float('inf')

    def get_expectation_values(self, params: np.ndarray, cost_terms: List[Tuple]) -> List[float]:
        """Get expectation values using the quantum circuit with improved error handling."""
        try:
            values = self.quantum_circuit.get_expectation_values(params, cost_terms)
            if not isinstance(values, list) or len(values) != self.n_qubits:
                raise ValueError(f"Invalid expectation values: expected list of length {self.n_qubits}")
            return values
        except Exception as e:
            logger.error(f"Error getting expectation values: {str(e)}")
            return [0.0] * self.n_qubits