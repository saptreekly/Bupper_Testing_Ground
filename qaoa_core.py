import pennylane as qml
import numpy as np
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)

class QAOACircuit:
    def __init__(self, n_qubits: int, depth: int = 1):
        """Initialize QAOA circuit."""
        self.n_qubits = n_qubits
        self.depth = depth
        try:
            self.dev = qml.device('default.qubit', wires=n_qubits)
            self.circuit = qml.QNode(self._circuit_implementation, 
                                   self.dev,
                                   interface="autograd",
                                   diff_method="parameter-shift")
            logger.info("Initialized quantum device with %d qubits", n_qubits)
        except Exception as e:
            logger.error("Failed to initialize quantum device: %s", str(e))
            raise

    def _circuit_implementation(self, params, cost_terms):
        """Simplified QAOA circuit implementation."""
        try:
            # Initial state
            logger.debug("Preparing initial state with %d qubits", self.n_qubits)
            for i in range(self.n_qubits):
                qml.Hadamard(wires=i)

            # QAOA layers
            for layer in range(self.depth):
                # Problem unitary
                gamma = params[2*layer]
                for coeff, (i, j) in cost_terms:
                    if i != j:  # Skip self-interactions
                        qml.CNOT(wires=[i, j])
                        qml.RZ(2 * gamma * coeff, wires=j)
                        qml.CNOT(wires=[i, j])

                # Mixer unitary
                beta = params[2*layer + 1]
                for i in range(self.n_qubits):
                    qml.RX(2 * beta, wires=i)

            # Measure expectation values
            measurements = [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
            logger.debug("Circuit measurements shape: %d", len(measurements))
            return measurements

        except Exception as e:
            logger.error("Error in circuit implementation: %s", str(e))
            raise

    def optimize(self, cost_terms: List[Tuple], steps: int = 100):
        """Optimize QAOA parameters."""
        try:
            # Initialize parameters
            params = np.array([np.pi/4, np.pi/2] * self.depth)
            logger.info("Initial parameters: %s", str(params))

            # Use Adam optimizer with careful learning rate
            opt = qml.AdamOptimizer(stepsize=0.01)
            costs = []

            def cost_function(params):
                try:
                    # Get measurements
                    measurements = np.asarray(self.circuit(params, cost_terms))
                    logger.debug("Raw measurements: %s", str(measurements))

                    # Calculate cost
                    cost = 0.0
                    for coeff, (i, j) in cost_terms:
                        term = float(measurements[i]) * float(measurements[j]) * coeff
                        cost += term
                        logger.debug("Cost term (%d,%d): %.6f", i, j, term)

                    cost_val = float(np.asarray(cost))
                    logger.debug("Total cost: %.6f", cost_val)
                    return cost

                except Exception as e:
                    logger.error("Error in cost function: %s", str(e))
                    raise

            # Optimization loop
            prev_cost = float('inf')
            best_cost = float('inf')
            best_params = None

            for step in range(steps):
                try:
                    # Optimization step
                    params, current_cost = opt.step_and_cost(cost_function, params)

                    # Convert cost to float safely
                    cost_val = float(np.asarray(current_cost))
                    costs.append(cost_val)

                    # Track best solution
                    if cost_val < best_cost:
                        best_cost = cost_val
                        best_params = params.copy()
                        logger.info("New best solution - Step: %d, Cost: %.6f", step, best_cost)

                    # Convergence check
                    if abs(prev_cost - cost_val) < 1e-6:
                        logger.info("Converged at step %d", step)
                        break

                    prev_cost = cost_val
                    logger.info("Step %d: Cost = %.6f", step, cost_val)

                except Exception as e:
                    logger.error("Error in optimization step %d: %s", step, str(e))
                    raise

            return best_params if best_params is not None else params, costs

        except Exception as e:
            logger.error("Error during optimization: %s", str(e))
            raise

    def compute_expectation(self, optimal_params: np.ndarray, cost_terms: List[Tuple]) -> float:
        """Compute expectation value with optimal parameters."""
        try:
            measurements = np.asarray(self.circuit(optimal_params, cost_terms))
            cost = 0.0
            for coeff, (i, j) in cost_terms:
                cost += coeff * float(measurements[i]) * float(measurements[j])
            return float(np.asarray(cost))

        except Exception as e:
            logger.error("Error computing expectation: %s", str(e))
            raise