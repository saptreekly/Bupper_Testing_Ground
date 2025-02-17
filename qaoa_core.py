import pennylane as qml
import numpy as np
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)

class QAOACircuit:
    def __init__(self, n_qubits: int, depth: int = 1):
        """Initialize QAOA circuit with size limits."""
        self.n_qubits = min(n_qubits, 25)  # Hard limit at 25 qubits
        self.depth = depth
        try:
            # Use default.qubit with numpy interface
            self.dev = qml.device('default.qubit', wires=self.n_qubits)
            self.circuit = qml.QNode(self._circuit_implementation, 
                                 self.dev,
                                 interface="numpy")
            logger.info("Initialized quantum device with %d qubits", self.n_qubits)
        except Exception as e:
            logger.error("Failed to initialize quantum device: %s", str(e))
            raise

    def _validate_cost_terms(self, cost_terms: List[Tuple]) -> List[Tuple]:
        """Validate cost terms and ensure they are within qubit range."""
        valid_terms = []
        seen_pairs = set()
        for coeff, (i, j) in cost_terms:
            # Ensure indices are within range and not equal
            if (0 <= i < self.n_qubits and 
                0 <= j < self.n_qubits and 
                i != j and 
                (i, j) not in seen_pairs and 
                (j, i) not in seen_pairs):
                valid_terms.append((float(coeff), (i, j)))
                seen_pairs.add((i, j))

        if not valid_terms:
            logger.warning("No valid cost terms found!")
        else:
            logger.info(f"Using {len(valid_terms)} valid cost terms out of {len(cost_terms)}")

        return valid_terms

    def _circuit_implementation(self, params, cost_terms):
        """Basic QAOA circuit implementation."""
        try:
            # Initial state preparation
            for i in range(self.n_qubits):
                qml.Hadamard(wires=i)

            # Single layer of QAOA
            gamma = params[0]
            beta = params[1]

            # Cost Hamiltonian with validated terms
            valid_terms = self._validate_cost_terms(cost_terms)
            for coeff, (i, j) in valid_terms:
                qml.CNOT(wires=[i, j])
                qml.RZ(2 * gamma * coeff, wires=j)
                qml.CNOT(wires=[i, j])

            # Mixer Hamiltonian
            for i in range(self.n_qubits):
                qml.RX(2 * beta, wires=i)

            # Measure in computational basis
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        except Exception as e:
            logger.error("Error in circuit implementation: %s", str(e))
            raise

    def optimize(self, cost_terms: List[Tuple], steps: int = 10):
        """Basic QAOA optimization."""
        try:
            # Initialize parameters (scaled to avoid overflow)
            params = 0.01 * np.random.randn(2)
            opt = qml.GradientDescentOptimizer(stepsize=0.1)
            costs = []

            # Validate cost terms once before optimization
            valid_terms = self._validate_cost_terms(cost_terms)
            if not valid_terms:
                raise ValueError("No valid cost terms for optimization")

            def cost_function(p):
                """Basic cost function."""
                try:
                    # Get measurements
                    measurements = np.array(self.circuit(p, valid_terms))

                    # Calculate cost with validated terms
                    cost = 0.0
                    for coeff, (i, j) in valid_terms:
                        cost += coeff * measurements[i] * measurements[j]

                    # Scale cost for numerical stability
                    return 0.1 * float(cost)

                except Exception as e:
                    logger.error("Error in cost function: %s", str(e))
                    raise

            # Optimization loop
            for step in range(steps):
                try:
                    # Gradient descent step
                    params = opt.step(cost_function, params)

                    # Calculate and log current cost
                    current_cost = cost_function(params)
                    costs.append(float(current_cost))

                    logger.info("Step %d: Cost = %.6f, Params = [%.4f, %.4f]", 
                              step, current_cost, params[0], params[1])

                except Exception as e:
                    logger.error("Error in optimization step %d: %s", step, str(e))
                    break

            return params, costs

        except Exception as e:
            logger.error("Error during optimization: %s", str(e))
            raise