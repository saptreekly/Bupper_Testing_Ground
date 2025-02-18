import pennylane as qml
import numpy as np
from typing import List, Tuple, Optional, Callable
import logging

logger = logging.getLogger(__name__)

class QAOACircuit:
    def __init__(self, n_qubits: int, depth: int = 1, progress_callback: Optional[Callable] = None):
        """Initialize QAOA circuit."""
        self.n_qubits = n_qubits
        self.depth = depth
        self.progress_callback = progress_callback

        try:
            logger.info(f"Initializing QAOA with {n_qubits} qubits and depth {depth}")
            self.dev = qml.device('default.qubit', wires=n_qubits)
            logger.info("Successfully created quantum device")

            @qml.qnode(self.dev)
            def circuit(params, cost_terms):
                logger.debug(f"Executing circuit with params={params}")
                # Initial state preparation
                for i in range(n_qubits):
                    qml.Hadamard(wires=i)

                gamma, beta = params

                # Cost Hamiltonian
                logger.debug("Applying cost Hamiltonian")
                for coeff, (i, j) in cost_terms:
                    if i != j:
                        qml.CNOT(wires=[i, j])
                        qml.RZ(2 * gamma * coeff, wires=j)
                        qml.CNOT(wires=[i, j])

                # Mixer Hamiltonian
                logger.debug("Applying mixer Hamiltonian")
                for i in range(n_qubits):
                    qml.RX(2 * beta, wires=i)

                logger.debug("Measuring expectation values")
                return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

            self.circuit = circuit
            logger.info("Successfully initialized quantum circuit")

        except Exception as e:
            logger.error(f"Failed to initialize QAOA circuit: {str(e)}", exc_info=True)
            raise

    def get_expectation_values(self, params: np.ndarray, cost_terms: List[Tuple]) -> np.ndarray:
        """Compute expectation values."""
        try:
            logger.debug(f"Computing expectation values with params={params}")
            params = self._validate_params(params)
            result = self.circuit(params, cost_terms)
            logger.debug(f"Expectation values computed: {result}")
            return result
        except Exception as e:
            logger.error(f"Error computing expectation values: {str(e)}", exc_info=True)
            raise

    def optimize(self, cost_terms: List[Tuple], **kwargs) -> Tuple[np.ndarray, List[float]]:
        """Optimize QAOA parameters."""
        try:
            logger.info("Starting QAOA optimization")
            # Initialize parameters with slightly randomized values
            params = np.array([0.01, np.pi/4])  # Start with simpler initial values
            costs = []
            steps = kwargs.get('steps', 10)  # Default to 10 steps

            def cost_function(p):
                try:
                    logger.debug(f"Evaluating cost function for params={p}")
                    measurements = self.get_expectation_values(p, cost_terms)

                    # Calculate energy using proper quantum expectation values
                    energy = 0.0
                    for coeff, (i, j) in cost_terms:
                        # Calculate correlation between qubits i and j
                        energy += coeff * measurements[i] * measurements[j]

                    logger.debug(f"Cost function value: {energy}")
                    return float(energy)
                except Exception as e:
                    logger.error(f"Error in cost function: {str(e)}", exc_info=True)
                    return float('inf')

            # Use fixed learning rate and momentum for stability
            learning_rate = 0.1
            momentum = 0.5
            velocity = np.zeros_like(params)

            current_cost = cost_function(params)
            costs.append(current_cost)
            logger.info(f"Initial cost: {current_cost}")

            best_cost = current_cost
            best_params = params.copy()
            no_improvement_count = 0

            for step in range(steps):
                try:
                    logger.debug(f"Optimization step {step+1}/{steps}")

                    # Compute gradient using central differences
                    grad = np.zeros(2)
                    eps = 1e-3  # Larger epsilon for numerical stability

                    for i in range(2):
                        params_plus = params.copy()
                        params_plus[i] += eps
                        params_minus = params.copy()
                        params_minus[i] -= eps

                        cost_plus = cost_function(params_plus)
                        cost_minus = cost_function(params_minus)

                        if cost_plus != float('inf') and cost_minus != float('inf'):
                            grad[i] = (cost_plus - cost_minus) / (2 * eps)
                        else:
                            grad[i] = 0  # Skip update if cost computation failed

                    # Gradient norm for stopping condition
                    grad_norm = np.linalg.norm(grad)
                    if grad_norm > 1e-8:  # Only update if gradient is significant
                        # Update velocity and parameters with momentum
                        velocity = momentum * velocity - learning_rate * (grad / grad_norm)
                        params += velocity
                        params = self._validate_params(params)

                        # Compute new cost
                        new_cost = cost_function(params)
                        costs.append(new_cost)

                        # Update best solution if improved
                        if new_cost < best_cost:
                            best_cost = new_cost
                            best_params = params.copy()
                            no_improvement_count = 0
                        else:
                            no_improvement_count += 1

                        logger.info(f"Step {step+1}: cost={new_cost:.6f}, params={params}, grad_norm={grad_norm:.6f}")

                        if self.progress_callback:
                            self.progress_callback(step, {
                                "status": "Optimizing quantum circuit",
                                "progress": (step + 1) / steps,
                                "step": step,
                                "total_steps": steps,
                                "cost": float(new_cost)
                            })

                    else:
                        logger.info(f"Gradient too small ({grad_norm:.2e}), stopping optimization")
                        break

                    # Early stopping if no improvement
                    if no_improvement_count > 5:
                        logger.info("Early stopping due to no improvement")
                        break

                except Exception as e:
                    logger.error(f"Error in optimization step {step}: {str(e)}", exc_info=True)
                    continue

            logger.info(f"Optimization completed. Best cost: {best_cost:.6f}")
            return best_params, costs

        except Exception as e:
            logger.error(f"Error during optimization: {str(e)}", exc_info=True)
            raise

    def _validate_params(self, params: np.ndarray) -> np.ndarray:
        """Validate and normalize parameters."""
        try:
            logger.debug(f"Validating parameters: {params}")
            if not isinstance(params, np.ndarray):
                params = np.array(params)

            if params.size != 2:
                logger.warning(f"Invalid parameter size: {params.size}, resetting to default")
                params = np.array([0.01, np.pi/4])

            # Clip parameters to reasonable ranges
            params[0] = np.clip(params[0], -2*np.pi, 2*np.pi)  # gamma
            params[1] = np.clip(params[1], -np.pi, np.pi)      # beta

            logger.debug(f"Validated parameters: {params}")
            return params

        except Exception as e:
            logger.error(f"Error in parameter validation: {str(e)}", exc_info=True)
            raise