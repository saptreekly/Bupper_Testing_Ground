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
            # Use default.qubit with autograd interface for better gradient computation
            self.dev = qml.device('default.qubit', wires=self.n_qubits)
            self.circuit = qml.QNode(self._circuit_implementation, 
                                   self.dev,
                                   interface="autograd",
                                   diff_method="parameter-shift")
            logger.info("Initialized quantum device with %d qubits", self.n_qubits)
        except Exception as e:
            logger.error("Failed to initialize quantum device: %s", str(e))
            raise

    def _validate_cost_terms(self, cost_terms: List[Tuple]) -> List[Tuple]:
        """Validate and normalize cost terms with relaxed filtering."""
        valid_terms = []
        seen_wire_pairs = set()

        # First pass: find maximum coefficient for scaling
        max_coeff = max(abs(coeff) for coeff, _ in cost_terms) if cost_terms else 1.0
        logger.debug("Maximum coefficient: %.6f", max_coeff)

        # Relaxed threshold (0.01% of max coefficient)
        threshold = max_coeff * 0.0001
        logger.debug("Coefficient threshold: %.6f", threshold)

        for coeff, (i, j) in cost_terms:
            if not (0 <= i < self.n_qubits and 0 <= j < self.n_qubits):
                logger.debug("Skipping term with out-of-range wires: (%d, %d)", i, j)
                continue

            if i == j:
                logger.debug("Skipping self-interaction term: wire %d", i)
                continue

            wire_pair = tuple(sorted([i, j]))
            if wire_pair in seen_wire_pairs:
                logger.debug("Skipping duplicate wire pair: %s", wire_pair)
                continue

            # Normalize coefficient for numerical stability
            norm_coeff = coeff / max_coeff
            valid_terms.append((norm_coeff, wire_pair))
            seen_wire_pairs.add(wire_pair)
            logger.debug("Added term: coeff=%.6f, wires=%s", norm_coeff, wire_pair)

        if not valid_terms:
            logger.warning("No valid cost terms found after validation!")
        else:
            logger.info("Using %d valid cost terms out of %d original terms", 
                       len(valid_terms), len(cost_terms))

        return valid_terms

    def _circuit_implementation(self, params, cost_terms):
        """QAOA circuit implementation using IsingXX gates with multiple layers."""
        try:
            # Convert params to array for proper gradient computation
            params = np.array(params, requires_grad=True)

            # Initial state preparation
            for i in range(self.n_qubits):
                qml.Hadamard(wires=i)

            # Implement QAOA layers
            for layer in range(self.depth):
                # Cost Hamiltonian with IsingXX interactions
                gamma = params[2 * layer]
                beta = params[2 * layer + 1]

                # Apply cost Hamiltonian
                for coeff, (i, j) in cost_terms:
                    qml.IsingXX(2 * gamma * coeff, wires=[i, j])

                # Apply mixer Hamiltonian
                for i in range(self.n_qubits):
                    qml.RX(2 * beta, wires=i)

            # Measure in computational basis
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        except Exception as e:
            logger.error("Error in circuit implementation: %s", str(e))
            raise

    def optimize(self, cost_terms: List[Tuple], steps: int = 100):
        """Enhanced QAOA optimization with better parameter initialization."""
        try:
            valid_terms = self._validate_cost_terms(cost_terms)
            if not valid_terms:
                raise ValueError("No valid cost terms for optimization")

            # Initialize parameters with proper scaling
            params = np.random.uniform(low=-np.pi/4, high=np.pi/4, size=2 * self.depth)
            params = np.array(params, requires_grad=True)
            logger.info("Initial parameters: %s", str(params))

            # Use Adam optimizer for better convergence
            opt = qml.AdamOptimizer(stepsize=0.1)
            costs = []

            def cost_function(p):
                """Enhanced cost function with proper scaling."""
                try:
                    measurements = self.circuit(p, valid_terms)
                    cost = 0.0
                    for coeff, (i, j) in valid_terms:
                        term_cost = coeff * measurements[i] * measurements[j]
                        cost += term_cost
                        logger.debug("Term (%d,%d) contribution: %.6f", i, j, term_cost)

                    final_cost = float(cost)
                    logger.debug("Final cost: %.6f", final_cost)
                    return final_cost

                except Exception as e:
                    logger.error("Error in cost function: %s", str(e))
                    raise

            # Optimization loop with early stopping
            best_cost = float('inf')
            patience = 10
            no_improvement = 0

            for step in range(steps):
                try:
                    # Explicitly compute gradients and update parameters
                    params = opt.step(cost_function, params)
                    current_cost = cost_function(params)
                    costs.append(float(current_cost))

                    logger.info("Step %d: Cost = %.6f, Params = %s", 
                              step, current_cost, str(params))

                    # Early stopping logic
                    if current_cost < best_cost - 1e-6:  # Small threshold for numerical stability
                        best_cost = current_cost
                        no_improvement = 0
                    else:
                        no_improvement += 1
                        if no_improvement >= patience:
                            logger.info("Early stopping triggered after %d steps", step)
                            break

                except Exception as e:
                    logger.error("Error in optimization step %d: %s", step, str(e))
                    break

            return params, costs

        except Exception as e:
            logger.error("Error during optimization: %s", str(e))
            raise