import pennylane as qml
import numpy as np
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)

class QAOACircuit:
    def __init__(self, n_qubits: int, depth: int = 1):
        """Initialize QAOA circuit with size limits and adaptive depth."""
        # Hard limit at 25 qubits for PennyLane
        self.max_qubits = 25
        if n_qubits > self.max_qubits:
            raise ValueError(f"Number of qubits ({n_qubits}) exceeds maximum allowed ({self.max_qubits})")

        self.n_qubits = min(n_qubits, self.max_qubits)
        # Adaptive depth based on problem size
        self.depth = min(depth, max(1, self.n_qubits // 4))
        try:
            self.dev = qml.device('default.qubit', wires=self.n_qubits)
            self.circuit = qml.QNode(self._circuit_implementation, 
                                   self.dev,
                                   interface="autograd",
                                   diff_method="backprop")
            logger.info("Initialized quantum device with %d qubits and depth %d", 
                      self.n_qubits, self.depth)
        except Exception as e:
            logger.error("Failed to initialize quantum device: %s", str(e))
            raise

    def _validate_cost_terms(self, cost_terms: List[Tuple]) -> List[Tuple]:
        """Validate and normalize cost terms with improved numerical stability."""
        valid_terms = []
        seen_wire_pairs = set()

        if not cost_terms:
            logger.warning("Empty cost terms provided")
            return []

        # Calculate statistics for robust scaling
        coeffs = [abs(coeff) for coeff, _ in cost_terms]
        max_coeff = max(coeffs)
        mean_coeff = sum(coeffs) / len(coeffs)
        threshold = mean_coeff * 0.01  # Adaptive threshold based on mean

        logger.debug("Cost terms statistics: max=%.6f, mean=%.6f, threshold=%.6f",
                    max_coeff, mean_coeff, threshold)

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

            if abs(coeff) > threshold:
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
        """QAOA circuit implementation with improved error handling."""
        try:
            # Initial state preparation with noise resilience
            for i in range(self.n_qubits):
                qml.Hadamard(wires=i)

            # Implement QAOA layers with error checking
            for layer in range(self.depth):
                # Cost Hamiltonian phase with parameter bounds
                gamma = np.clip(params[2 * layer], -2*np.pi, 2*np.pi)
                for coeff, (i, j) in cost_terms:
                    qml.CNOT(wires=[i, j])
                    qml.RZ(2 * gamma * coeff, wires=j)
                    qml.CNOT(wires=[i, j])

                # Mixer Hamiltonian phase with bounded parameters
                beta = np.clip(params[2 * layer + 1], -np.pi, np.pi)
                for i in range(self.n_qubits):
                    qml.RX(2 * beta, wires=i)

            # Return expectation values using computational basis
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        except Exception as e:
            logger.error("Error in circuit implementation: %s", str(e))
            raise

    def optimize(self, cost_terms: List[Tuple], steps: int = 100):
        """Enhanced QAOA optimization with improved convergence."""
        try:
            valid_terms = self._validate_cost_terms(cost_terms)
            if not valid_terms:
                raise ValueError("No valid cost terms for optimization")

            # Improved parameter initialization strategy
            params = []
            for _ in range(self.depth):
                # Initialize gamma near zero and beta near π/2 for better initial state
                params.extend([
                    np.random.uniform(-0.1, 0.1),  # gamma
                    np.random.uniform(np.pi/2 - 0.1, np.pi/2 + 0.1)  # beta
                ])
            params = np.array(params)
            logger.info("Initial parameters: %s", str(params))

            # Use Adam optimizer with adaptive learning rate
            opt = qml.AdamOptimizer(stepsize=0.05, beta1=0.9, beta2=0.999)
            costs = []
            best_cost = float('inf')
            best_params = None
            no_improvement_count = 0
            min_improvement = 1e-4

            def cost_function(p):
                """Enhanced cost function with proper scaling."""
                try:
                    measurements = self.circuit(p, valid_terms)
                    cost = sum(coeff * measurements[i] * measurements[j] 
                             for coeff, (i, j) in valid_terms)
                    return float(cost)
                except Exception as e:
                    logger.error("Error in cost function: %s", str(e))
                    raise

            # Optimization loop with early stopping
            for step in range(steps):
                try:
                    params, cost = opt.step_and_cost(cost_function, params)
                    costs.append(float(cost))

                    # Update best solution
                    if cost < best_cost - min_improvement:
                        best_cost = cost
                        best_params = params.copy()
                        no_improvement_count = 0
                    else:
                        no_improvement_count += 1

                    # Early stopping check
                    if no_improvement_count >= 10:
                        logger.info("Early stopping triggered at step %d", step)
                        break

                    if step % 10 == 0:
                        logger.info("Step %d: Cost = %.6f", step, cost)

                except Exception as e:
                    logger.error("Error in optimization step %d: %s", step, str(e))
                    break

            return best_params if best_params is not None else params, costs

        except Exception as e:
            logger.error("Error during optimization: %s", str(e))
            raise

    def get_expectation_values(self, params, cost_terms):
        """Get expectation values with improved error handling."""
        try:
            valid_terms = self._validate_cost_terms(cost_terms)
            if not valid_terms:
                raise ValueError("No valid cost terms for expectation value calculation")
            return self.circuit(params, valid_terms)
        except Exception as e:
            logger.error("Error getting expectation values: %s", str(e))
            raise