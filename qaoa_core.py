import pennylane as qml
import numpy as np
from typing import List, Tuple
import logging
import time

logger = logging.getLogger(__name__)

class QAOACircuit:
    def __init__(self, n_qubits: int, depth: int = 1):
        """Initialize QAOA circuit with size limits."""
        self.n_qubits = min(n_qubits, 25)  # Hard limit at 25 qubits
        self.depth = depth
        try:
            self.dev = qml.device('default.qubit', wires=self.n_qubits)
            self.circuit = qml.QNode(self._circuit_implementation, 
                                   self.dev,
                                   interface="autograd")
            logger.info("Initialized quantum device with %d qubits", self.n_qubits)
        except Exception as e:
            logger.error("Failed to initialize quantum device: %s", str(e))
            raise

    def _circuit_implementation(self, params, cost_terms):
        """Simplified QAOA circuit with minimal operations."""
        try:
            active_qubits = {i for term in cost_terms for _, (i, j) in [term] for i in [i, j]}
            logger.debug("Active qubits: %s", sorted(list(active_qubits)))

            # Initial state preparation
            for i in active_qubits:
                qml.Hadamard(wires=i)

            # Cost unitary
            for coeff, (i, j) in cost_terms:
                if i != j:
                    qml.CNOT(wires=[i, j])
                    qml.RZ(params[0] * coeff * 0.1, wires=j)
                    qml.CNOT(wires=[i, j])

            # Mixer unitary
            for i in active_qubits:
                qml.RX(params[1], wires=i)

            # Return measurements for all qubits
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        except Exception as e:
            logger.error("Error in circuit implementation: %s", str(e))
            raise

    def optimize(self, cost_terms: List[Tuple], steps: int = 10):
        """Minimalist QAOA optimization."""
        try:
            # Filter significant terms
            threshold = 0.15  # Higher threshold for stability
            significant_terms = [(coeff, (i, j)) for coeff, (i, j) in cost_terms 
                               if abs(coeff) > threshold]
            logger.info("Using %d of %d cost terms", len(significant_terms), len(cost_terms))

            # Initialize parameters using PennyLane's array interface
            params = qml.numpy.array([0.1, 0.1], requires_grad=True)
            opt = qml.GradientDescentOptimizer(stepsize=0.05)
            costs = []

            best_params = None
            best_cost = float('inf')
            no_improvement_count = 0

            # Define cost function using quantum measurements
            def cost_function(params):
                measurements = self.circuit(params, significant_terms)
                cost = sum(0.1 * coeff * measurements[i] * measurements[j]
                          for coeff, (i, j) in significant_terms)
                return cost

            # Run optimization loop
            for step in range(steps):
                try:
                    # Optimization step
                    params, cost = opt.step_and_cost(cost_function, params)
                    cost_val = float(cost)
                    costs.append(cost_val)

                    logger.info("Step %d: Cost = %.6f", step, cost_val)

                    # Update best parameters
                    if cost_val < best_cost:
                        best_cost = cost_val
                        best_params = params.copy()
                        no_improvement_count = 0
                    else:
                        no_improvement_count += 1

                    # Early stopping
                    if no_improvement_count >= 3:
                        logger.info("Early stopping due to no improvement")
                        break

                except Exception as e:
                    logger.error("Error in optimization step %d: %s", step, str(e))
                    break

            return best_params if best_params is not None else params, costs

        except Exception as e:
            logger.error("Error during optimization: %s", str(e))
            raise