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

            # Single QAOA layer
            gamma, beta = params[0], params[1]
            logger.debug("QAOA parameters: gamma=%s, beta=%s", str(gamma), str(beta))

            # Problem unitary (ZZ interactions)
            logger.debug("Applying problem unitary with %d cost terms", len(cost_terms))
            for coeff, (i, j) in cost_terms:
                if i != j:  # Skip self-interactions
                    qml.CNOT(wires=[i, j])
                    qml.RZ(2 * gamma * coeff, wires=j)
                    qml.CNOT(wires=[i, j])

            # Mixer unitary
            logger.debug("Applying mixer unitary")
            for i in range(self.n_qubits):
                qml.RX(2 * beta, wires=i)

            # Measure expectation values
            measurements = [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
            logger.debug("Circuit measurements shape: %d", len(measurements))
            return measurements

        except Exception as e:
            logger.error("Error in circuit implementation: %s", str(e))
            raise

    def optimize(self, cost_terms: List[Tuple], steps: int = 20):
        """Optimize QAOA parameters with debugging."""
        try:
            # Initialize parameters with better starting point
            params = qml.numpy.array([qml.numpy.pi/4, qml.numpy.pi/2], requires_grad=True)
            logger.info("Initial parameters: %s", str(params))

            # Use Adam optimizer with smaller learning rate
            opt = qml.AdamOptimizer(stepsize=0.01)
            costs = []

            def cost_function(params):
                try:
                    # Get measurements and verify dimensions
                    measurements = self.circuit(params, cost_terms)
                    measurements = qml.numpy.array(measurements)
                    assert len(measurements) == self.n_qubits, f"Expected {self.n_qubits} measurements, got {len(measurements)}"
                    logger.debug("Measurements: %s", str(measurements))

                    # Calculate cost with normalization
                    cost = 0.0
                    n_terms = 0
                    for coeff, (i, j) in cost_terms:
                        term_cost = coeff * measurements[i] * measurements[j]
                        logger.debug("Cost term (%d,%d) with coeff %.3f: %s", 
                                i, j, coeff, str(term_cost))
                        cost += term_cost
                        n_terms += 1

                    # Normalize cost by number of terms
                    if n_terms > 0:
                        cost = cost / n_terms

                    logger.debug("Total normalized cost: %s", str(cost))
                    return cost

                except Exception as e:
                    logger.error("Error in cost function: %s", str(e))
                    raise

            prev_cost = float('inf')
            for step in range(steps):
                try:
                    # One optimization step with parameter update
                    params, current_cost = opt.step_and_cost(cost_function, params)

                    # Convert cost value for history
                    if hasattr(current_cost, 'numpy'):
                        history_cost = float(current_cost.numpy())
                    else:
                        history_cost = float(current_cost)

                    costs.append(history_cost)

                    # Log progress
                    logger.info("Iteration %d: Cost = %.6f", step, history_cost)

                    # Check convergence
                    if abs(prev_cost - history_cost) < 1e-6:
                        logger.info("Optimization converged within tolerance")
                        break

                    prev_cost = history_cost

                except Exception as e:
                    logger.error("Error in optimization step %d: %s", step, str(e))
                    raise

            return params, costs

        except Exception as e:
            logger.error("Error during optimization: %s", str(e))
            raise

    def compute_expectation(self, optimal_params: np.ndarray, 
                          cost_terms: List[Tuple]) -> float:
        """
        Compute expectation value with optimal parameters.
        """
        try:
            # Get measurements as numpy array
            measurements = np.array(self.circuit(optimal_params, cost_terms))

            # Calculate final cost using measurements
            cost = 0.0
            for coeff, (i, j) in cost_terms:
                cost += coeff * measurements[i] * measurements[j]

            if hasattr(cost, 'numpy'):
                cost = float(cost.numpy())
            return float(cost)

        except Exception as e:
            logger.error("Error computing expectation: %s", str(e))
            raise