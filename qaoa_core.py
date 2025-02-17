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
            # Strict validation of wire indices
            if not (0 <= i < self.n_qubits and 0 <= j < self.n_qubits):
                logger.debug("Skipping term with out-of-range wires: (%d, %d)", i, j)
                continue

            if i == j:
                logger.debug("Skipping self-interaction term: wire %d", i)
                continue

            # Ensure consistent ordering of wire pairs
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
        """QAOA circuit implementation using IsingXX gates."""
        try:
            # Initial state preparation
            for i in range(self.n_qubits):
                qml.Hadamard(wires=i)

            # Single layer of QAOA
            gamma = params[0]
            beta = params[1]

            # Cost Hamiltonian with IsingXX interactions
            for coeff, (i, j) in cost_terms:  # cost_terms are already validated
                qml.IsingXX(2 * gamma * coeff, wires=[i, j])  # Factor of 2 for XX convention

            # Mixer Hamiltonian (kept as RX gates)
            for i in range(self.n_qubits):
                qml.RX(2 * beta, wires=i)

            # Measure in computational basis
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        except Exception as e:
            logger.error("Error in circuit implementation: %s", str(e))
            raise

    def optimize(self, cost_terms: List[Tuple], steps: int = 10):
        """Basic QAOA optimization with adjusted parameters for XX gates."""
        try:
            # Validate cost terms once before optimization
            valid_terms = self._validate_cost_terms(cost_terms)
            if not valid_terms:
                raise ValueError("No valid cost terms for optimization")

            # Initialize parameters (scaled for XX gates)
            params = 0.01 * np.random.randn(2)
            logger.info("Initial parameters: gamma=%.6f, beta=%.6f", params[0], params[1])

            opt = qml.GradientDescentOptimizer(stepsize=0.1)
            costs = []

            def cost_function(p):
                """Basic cost function."""
                try:
                    measurements = np.array(self.circuit(p, valid_terms))
                    cost = 0.0
                    for coeff, (i, j) in valid_terms:
                        term_cost = coeff * measurements[i] * measurements[j]
                        cost += term_cost
                        logger.debug("Term (%d,%d) contribution: %.6f", i, j, term_cost)

                    # Scale cost for numerical stability
                    final_cost = 0.1 * float(cost)
                    logger.debug("Final cost: %.6f", final_cost)
                    return final_cost

                except Exception as e:
                    logger.error("Error in cost function: %s", str(e))
                    raise

            # Optimization loop
            for step in range(steps):
                try:
                    params = opt.step(cost_function, params)
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