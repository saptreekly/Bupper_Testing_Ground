import pennylane as qml
import numpy as np
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)

class QAOACircuit:
    def __init__(self, n_qubits: int, depth: int = 1):
        """
        Initialize QAOA circuit for routing optimization.
        """
        self.n_qubits = n_qubits
        self.depth = depth
        try:
            # Create quantum device with better gradient computation
            self.dev = qml.device('default.qubit', wires=n_qubits, shots=1000)
            self.circuit = qml.QNode(self._circuit_implementation, 
                                   self.dev,
                                   interface="autograd",
                                   diff_method="parameter-shift")
            logger.info("Successfully initialized quantum device with %d qubits", n_qubits)
        except Exception as e:
            logger.error("Failed to initialize quantum device: %s", str(e))
            raise

    def _circuit_implementation(self, params, cost_terms):
        """
        Implementation of the QAOA circuit.
        """
        try:
            # Initialize in superposition
            for i in range(self.n_qubits):
                qml.Hadamard(wires=i)
                logger.debug("Applied Hadamard to qubit %d", i)

            # Apply QAOA layers
            for layer in range(self.depth):
                gamma = params[2 * layer]
                beta = params[2 * layer + 1]

                logger.debug("Layer %d: gamma = %s, beta = %s", layer, str(gamma), str(beta))

                # Cost unitary
                for coeff, pauli_terms in cost_terms:
                    ham = self._create_pauli_product(pauli_terms)
                    qml.ApproxTimeEvolution(ham, gamma * coeff, 1)

                # Mixer unitary
                for i in range(self.n_qubits):
                    qml.RX(2 * beta, wires=i)

            # Return measurements for all qubits
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        except Exception as e:
            logger.error("Error in circuit execution: %s", str(e))
            raise

    def _create_pauli_product(self, pauli_terms):
        """Helper function to create a product of Pauli operators"""
        try:
            ops = []
            for term in pauli_terms:
                wire = int(term[1:])
                if term.startswith('Z'):
                    ops.append(qml.PauliZ(wire))
                elif term.startswith('X'):
                    ops.append(qml.PauliX(wire))
                elif term.startswith('Y'):
                    ops.append(qml.PauliY(wire))

            if not ops:
                return qml.Identity(0)

            # Use tensor product operator directly
            result = ops[0]
            for op in ops[1:]:
                result = result @ op
            return result

        except Exception as e:
            logger.error("Error creating Pauli product: %s", str(e))
            raise

    def cost_hamiltonian(self, adjacency_matrix: np.ndarray) -> List[Tuple]:
        """
        Construct cost Hamiltonian terms from adjacency matrix.
        """
        terms = []
        try:
            for i in range(self.n_qubits):
                for j in range(i + 1, self.n_qubits):
                    if adjacency_matrix[i, j] != 0:
                        terms.append((float(adjacency_matrix[i, j]), [f"Z{i}", f"Z{j}"]))
            logger.info("Created %d cost Hamiltonian terms", len(terms))
            return terms
        except Exception as e:
            logger.error("Error in cost_hamiltonian: %s", str(e))
            raise

    def mixer_hamiltonian(self) -> List[Tuple]:
        """
        Construct mixer Hamiltonian terms.
        """
        return [(1.0, [f"X{i}"]) for i in range(self.n_qubits)]

    def optimize(self, cost_terms: List[Tuple], steps: int = 100):
        """
        Optimize QAOA parameters.
        """
        try:
            # Initialize parameters with better initial values
            params = qml.numpy.array([0.01, np.pi/4] * self.depth, requires_grad=True)

            def objective(params):
                measurements = self.circuit(params, cost_terms)
                # Calculate cost using individual measurements
                cost = sum(coeff * measurements[i] * measurements[j]
                          for coeff, (i, j) in cost_terms)
                return cost

            # Use smaller learning rate for better convergence
            opt = qml.AdamOptimizer(stepsize=0.01)
            costs = []

            for step in range(steps):
                params, cost = opt.step_and_cost(objective, params)
                cost_val = float(cost)
                costs.append(cost_val)

                if step % 5 == 0:  # Log more frequently
                    logger.info("Step %d: Cost = %s", step, str(cost_val))

            final_cost = costs[-1]
            logger.info("Optimization completed. Final cost: %s", str(final_cost))
            return params, costs

        except Exception as e:
            logger.error("Error in optimization: %s", str(e))
            raise