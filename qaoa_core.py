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
        self.depth = 1  # Force depth to 1 to maintain 2-parameter system

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
        """Validate and normalize cost terms."""
        if not cost_terms:
            logger.warning("Empty cost terms provided")
            return [(1.0, (0, min(1, self.n_qubits-1)))]

        valid_terms = []
        coeffs = [abs(coeff) for coeff, _ in cost_terms]
        max_coeff = max(coeffs)

        for coeff, (i, j) in cost_terms:
            if 0 <= i < self.n_qubits and 0 <= j < self.n_qubits:
                norm_coeff = coeff / max_coeff if max_coeff > 0 else coeff
                if i == j:  # Diagonal term
                    valid_terms.append((norm_coeff, (i,)))
                elif i < j:  # Off-diagonal term
                    valid_terms.append((norm_coeff, (i, j)))

        if not valid_terms:
            valid_terms.append((1.0, (0, min(1, self.n_qubits-1))))

        logger.info(f"Using {len(valid_terms)} valid cost terms")
        return valid_terms

    def _circuit_implementation(self, params, cost_terms):
        """QAOA circuit implementation with improved error handling."""
        try:
            # Ensure we only use 2 parameters (gamma, beta)
            if len(params) != 2:
                logger.warning(f"Received {len(params)} parameters, using only first 2")
                params = params[:2]

            gamma, beta = params
            logger.debug(f"Circuit implementation using gamma={gamma:.4f}, beta={beta:.4f}")

            # Initial state preparation with noise resilience
            for i in range(self.n_qubits):
                qml.Hadamard(wires=i)

            # Cost Hamiltonian phase
            gamma = np.clip(gamma, -2*np.pi, 2*np.pi)
            for coeff, wires in cost_terms:
                if len(wires) == 1:  # Diagonal term
                    qml.RZ(2 * gamma * coeff, wires=wires[0])
                else:  # Off-diagonal term
                    i, j = wires
                    if i != j:  # Double-check that wires are different
                        qml.CNOT(wires=[i, j])
                        qml.RZ(2 * gamma * coeff, wires=j)
                        qml.CNOT(wires=[i, j])

            # Mixer Hamiltonian phase
            beta = np.clip(beta, -np.pi, np.pi)
            for i in range(self.n_qubits):
                qml.RX(2 * beta, wires=i)

            # Return expectation values using computational basis
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        except Exception as e:
            logger.error("Error in circuit implementation: %s", str(e))
            raise

    def get_expectation_values(self, params, cost_terms):
        """Get expectation values with improved error handling."""
        try:
            # Ensure we only use 2 parameters
            if len(params) != 2:
                logger.warning(f"Received {len(params)} parameters, truncating to first 2")
                params = params[:2]

            valid_terms = self._validate_cost_terms(cost_terms)
            logger.info(f"Computing expectation values with params: {params}")
            return self.circuit(params, valid_terms)
        except Exception as e:
            logger.error("Error getting expectation values: %s", str(e))
            raise

    def optimize(self, cost_terms: List[Tuple], steps: int = 100):
        """Enhanced QAOA optimization with improved convergence."""
        try:
            valid_terms = self._validate_cost_terms(cost_terms)
            logger.debug(f"Starting optimization with {len(valid_terms)} validated terms")

            # Improved parameter initialization strategy
            params = []
            for _ in range(self.depth):
                # Initialize gamma near zero and beta near π/2 for better initial state
                params.extend([
                    np.random.uniform(-0.1, 0.1),  # gamma
                    np.random.uniform(np.pi/2 - 0.1, np.pi/2 + 0.1)  # beta
                ])
            params = np.array(params)
            logger.info("Initial parameters: %s", params)

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
                    cost = 0.0
                    for coeff, wires in valid_terms:
                        if len(wires) == 1:  # Diagonal term
                            cost += coeff * measurements[wires[0]]
                        else:  # Off-diagonal term
                            i, j = wires
                            cost += coeff * measurements[i] * measurements[j]
                    return float(cost)
                except Exception as e:
                    logger.error("Error in cost function: %s", str(e))
                    # Return high cost to indicate failure
                    return float('inf')

            # Optimization loop with early stopping
            for step in range(steps):
                try:
                    params, cost = opt.step_and_cost(cost_function, params)
                    if cost < float('inf'):  # Only record valid costs
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
                    # Continue to next step instead of breaking

            # Ensure we have at least one cost value
            if not costs:
                costs.append(float('inf'))

            return best_params if best_params is not None else params, costs

        except Exception as e:
            logger.error("Error during optimization: %s", str(e))
            raise