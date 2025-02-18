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
            # Initialize parameters
            params = np.array([0.01, np.pi/4])
            costs = []
            steps = kwargs.get('steps', 10)

            def cost_function(p):
                try:
                    logger.debug(f"Evaluating cost function for params={p}")
                    measurements = self.get_expectation_values(p, cost_terms)
                    total_cost = 0.0
                    for coeff, (i, j) in cost_terms:
                        total_cost += coeff * measurements[i] * measurements[j]
                    logger.debug(f"Cost function value: {total_cost}")
                    return float(total_cost)
                except Exception as e:
                    logger.error(f"Error in cost function: {str(e)}", exc_info=True)
                    return float('inf')

            # Simple gradient descent
            learning_rate = 0.1
            current_cost = cost_function(params)
            costs.append(current_cost)
            logger.info(f"Initial cost: {current_cost}")

            for step in range(steps):
                try:
                    logger.debug(f"Optimization step {step+1}/{steps}")
                    # Compute gradient
                    grad = np.zeros(2)
                    eps = 1e-4

                    for i in range(2):
                        params_plus = params.copy()
                        params_plus[i] += eps
                        cost_plus = cost_function(params_plus)
                        grad[i] = (cost_plus - current_cost) / eps

                    # Update parameters
                    params -= learning_rate * grad
                    params = self._validate_params(params)

                    # Compute new cost
                    new_cost = cost_function(params)
                    costs.append(new_cost)
                    current_cost = new_cost

                    logger.info(f"Step {step+1}: cost={current_cost:.6f}, params={params}")

                    if self.progress_callback:
                        self.progress_callback(step, {
                            "status": "Optimizing quantum circuit",
                            "progress": (step + 1) / steps,
                            "step": step,
                            "total_steps": steps,
                            "cost": float(current_cost)
                        })

                except Exception as e:
                    logger.error(f"Error in optimization step {step}: {str(e)}", exc_info=True)
                    continue

            logger.info(f"Optimization completed. Final cost: {current_cost:.6f}")
            return params, costs

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