import pennylane as qml
import numpy as np
from typing import List, Tuple
import logging
import time

logger = logging.getLogger(__name__)

class QAOACircuit:
    def __init__(self, n_qubits: int, depth: int = 2):
        """Initialize QAOA circuit."""
        self.n_qubits = n_qubits
        self.depth = depth
        try:
            # Check if number of qubits is too large
            if n_qubits > 20:  # 2^20 = 1,048,576 amplitudes
                logger.warning("Large number of qubits (%d) may cause memory issues", n_qubits)
                raise ValueError(f"Number of qubits ({n_qubits}) exceeds maximum supported (20)")

            # Use default.qubit for better memory efficiency
            self.dev = qml.device('default.qubit', wires=n_qubits)
            self.circuit = qml.QNode(self._circuit_implementation, 
                                   self.dev,
                                   interface="autograd")
            logger.info("Initialized quantum device with %d qubits", n_qubits)
        except Exception as e:
            logger.error("Failed to initialize quantum device: %s", str(e))
            raise

    def _circuit_implementation(self, params, cost_terms):
        """Simplified QAOA circuit implementation."""
        try:
            # Initial state preparation - superposition
            for i in range(self.n_qubits):
                qml.Hadamard(wires=i)

            # QAOA layers with simplified mixing
            for layer in range(self.depth):
                # Cost unitary
                gamma = params[2*layer]
                for coeff, (i, j) in cost_terms:
                    if i != j:
                        qml.CNOT(wires=[i, j])
                        qml.RZ(2 * gamma * coeff, wires=j)
                        qml.CNOT(wires=[i, j])

                # Mixer unitary
                beta = params[2*layer + 1]
                for i in range(self.n_qubits):
                    qml.RX(2 * beta, wires=i)

            # Return expectation values
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        except Exception as e:
            logger.error("Error in circuit implementation: %s", str(e))
            raise

    def optimize(self, cost_terms: List[Tuple], steps: int = 100, timeout: int = 300):
        """Optimize QAOA parameters with timeout."""
        try:
            # Initialize parameters with better scaling
            gamma_init = np.linspace(0.1, np.pi/2, self.depth)  # Narrower range for gamma
            beta_init = np.linspace(0.1, np.pi/4, self.depth)   # Narrower range for beta
            init_params = [val for pair in zip(gamma_init, beta_init) for val in pair]
            params = qml.numpy.array(init_params, requires_grad=True)

            # Use Adam optimizer with adaptive learning rate
            opt = qml.AdamOptimizer(stepsize=0.1)
            costs = []

            start_time = time.time()

            def cost_function(params):
                try:
                    measurements = self.circuit(params, cost_terms)
                    cost = 0.0
                    for coeff, (i, j) in cost_terms:
                        cost += coeff * measurements[i] * measurements[j]
                    return cost
                except Exception as e:
                    logger.error("Error in cost function: %s", str(e))
                    raise

            # Optimization loop with timeout
            best_cost = float('inf')
            best_params = None
            patience = 10
            patience_counter = 0

            for step in range(steps):
                if time.time() - start_time > timeout:
                    logger.info("Optimization timeout reached after %d steps", step)
                    break

                try:
                    params, cost = opt.step_and_cost(cost_function, params)
                    cost_val = float(cost)
                    costs.append(cost_val)

                    if cost_val < best_cost - 1e-3:
                        best_cost = cost_val
                        best_params = params.copy()
                        patience_counter = 0
                        logger.info("Step %d: Cost = %.6f", step, cost_val)
                    else:
                        patience_counter += 1

                    if patience_counter >= patience and step >= 30:
                        logger.info("Converged at step %d", step)
                        break

                except Exception as e:
                    logger.error("Error in optimization step %d: %s", step, str(e))
                    raise

            return best_params if best_params is not None else params, costs

        except Exception as e:
            logger.error("Error during optimization: %s", str(e))
            raise

    def compute_expectation(self, optimal_params: np.ndarray, 
                          cost_terms: List[Tuple]) -> float:
        """Compute expectation value with optimal parameters."""
        try:
            measurements = self.circuit(optimal_params, cost_terms)
            cost = 0.0
            for coeff, (i, j) in cost_terms:
                cost += coeff * measurements[i] * measurements[j]

            if hasattr(cost, 'numpy'):
                cost = float(cost.numpy())
            return float(cost)

        except Exception as e:
            logger.error("Error computing expectation: %s", str(e))
            raise