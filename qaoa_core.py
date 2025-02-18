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
                """Create and execute the QAOA circuit."""
                # Initial state preparation - equal superposition
                for i in range(n_qubits):
                    qml.Hadamard(wires=i)

                # Validate cost terms before applying quantum operations
                valid_terms = []
                for coeff, (i, j) in cost_terms:
                    if i != j and 0 <= i < n_qubits and 0 <= j < n_qubits:
                        valid_terms.append((coeff, (i, j)))

                if not valid_terms:
                    logger.warning("No valid cost terms found")
                    return [0.0] * n_qubits

                # QAOA layers
                for layer in range(depth):
                    # Problem unitary
                    gamma = params[2 * layer]

                    # Apply two-qubit ZZ interactions using IsingZZ operation
                    for coeff, (i, j) in valid_terms:
                        try:
                            # Use IsingZZ operation instead of CNOT+RZ combination
                            qml.IsingZZ(gamma * coeff, wires=[i, j])
                        except Exception as e:
                            logger.error(f"Error applying IsingZZ between qubits {i} and {j}: {str(e)}")
                            continue

                    # Mixer unitary
                    beta = params[2 * layer + 1]
                    for i in range(n_qubits):
                        qml.RX(2 * beta, wires=i)

                # Measure all qubits in Z basis
                return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

            self.circuit = circuit
            logger.info("Successfully initialized quantum circuit")

        except Exception as e:
            logger.error(f"Failed to initialize QAOA circuit: {str(e)}", exc_info=True)
            raise

    def optimize(self, cost_terms: List[Tuple], **kwargs) -> Tuple[np.ndarray, List[float]]:
        """Optimize QAOA parameters."""
        try:
            logger.info("Starting QAOA optimization")

            # Initialize parameters for all QAOA layers
            n_params = 2 * self.depth  # gamma and beta for each layer
            params = np.zeros(n_params)

            # Set initial parameters with small random values
            for i in range(self.depth):
                params[2*i] = np.random.uniform(0.01, 0.1)  # gamma
                params[2*i + 1] = np.random.uniform(np.pi/8, np.pi/4)  # beta

            # Validate cost terms
            valid_cost_terms = []
            for coeff, (i, j) in cost_terms:
                if i != j and 0 <= i < self.n_qubits and 0 <= j < self.n_qubits:
                    valid_cost_terms.append((coeff, (i, j)))

            if not valid_cost_terms:
                raise ValueError("No valid cost terms found after filtering")

            costs = []
            steps = kwargs.get('steps', 10)

            def cost_function(p):
                try:
                    measurements = self.get_expectation_values(p, valid_cost_terms)
                    energy = 0.0
                    for coeff, (i, j) in valid_cost_terms:
                        energy += coeff * measurements[i] * measurements[j]
                    return float(energy)
                except Exception as e:
                    logger.error(f"Error in cost function: {str(e)}", exc_info=True)
                    return float('inf')

            # Initial cost
            current_cost = cost_function(params)
            best_cost = current_cost
            best_params = params.copy()
            costs.append(current_cost)

            # Optimization parameters
            learning_rate = 0.05  # Reduced learning rate for stability
            epsilon = 1e-4  # Smaller epsilon for more precise gradients

            for step in range(steps):
                try:
                    # Compute gradient using finite differences
                    grad = np.zeros(n_params)
                    for i in range(n_params):
                        params_plus = params.copy()
                        params_plus[i] += epsilon
                        params_minus = params.copy()
                        params_minus[i] -= epsilon

                        cost_plus = cost_function(params_plus)
                        cost_minus = cost_function(params_minus)

                        if cost_plus != float('inf') and cost_minus != float('inf'):
                            grad[i] = (cost_plus - cost_minus) / (2 * epsilon)

                    # Update parameters using gradient descent
                    grad_norm = np.linalg.norm(grad)
                    if grad_norm > 1e-8:
                        params -= learning_rate * (grad / grad_norm)

                        # Ensure parameters stay in reasonable ranges
                        for i in range(self.depth):
                            params[2*i] = np.clip(params[2*i], -2*np.pi, 2*np.pi)  # gamma
                            params[2*i + 1] = np.clip(params[2*i + 1], -np.pi, np.pi)  # beta

                        current_cost = cost_function(params)
                        costs.append(current_cost)

                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_params = params.copy()

                        logger.info(f"Step {step+1}: cost={current_cost:.6f}, grad_norm={grad_norm:.6f}")

                        if self.progress_callback:
                            self.progress_callback(step, {
                                "status": "Optimizing quantum circuit",
                                "progress": (step + 1) / steps,
                                "step": step,
                                "cost": float(current_cost)
                            })
                    else:
                        logger.info(f"Gradient too small ({grad_norm:.2e}), stopping optimization")
                        break

                except Exception as e:
                    logger.error(f"Error in optimization step {step}: {str(e)}", exc_info=True)
                    continue

            return best_params, costs

        except Exception as e:
            logger.error(f"Error during optimization: {str(e)}", exc_info=True)
            raise

    def get_expectation_values(self, params: np.ndarray, cost_terms: List[Tuple]) -> np.ndarray:
        """Execute quantum circuit and get expectation values."""
        try:
            return self.circuit(params, cost_terms)
        except Exception as e:
            logger.error(f"Error computing expectation values: {str(e)}", exc_info=True)
            raise