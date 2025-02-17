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
            self.dev = qml.device('default.qubit', wires=n_qubits)
            self.circuit = qml.QNode(self._circuit_implementation, 
                                   self.dev,
                                   interface="autograd",
                                   diff_method="parameter-shift")
            logger.info(f"Successfully initialized quantum device with {n_qubits} qubits")
        except Exception as e:
            logger.error(f"Failed to initialize quantum device: {str(e)}")
            raise

    def _circuit_implementation(self, params, cost_terms):
        """
        Implementation of the QAOA circuit.
        """
        try:
            # Initialize in superposition
            for i in range(self.n_qubits):
                qml.Hadamard(wires=i)

            # Apply QAOA layers
            for layer in range(self.depth):
                gamma = params[2 * layer]
                beta = params[2 * layer + 1]

                # Cost unitary
                for coeff, pauli_terms in cost_terms:
                    ham = self._create_pauli_product(pauli_terms)
                    qml.ApproxTimeEvolution(ham, gamma * coeff, 1)

                # Mixer unitary (parallel application)
                qml.templates.layers.SimplifiedTwoDesign([beta], wires=range(self.n_qubits))

            # Create cost Hamiltonian operator
            cost_op = sum(coeff * self._create_pauli_product(pauli_terms)
                         for coeff, pauli_terms in cost_terms)

            # Return expectation value
            return qml.expval(cost_op)

        except Exception as e:
            logger.error(f"Error in circuit execution: {str(e)}")
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

            # Use tensor product for multiple operators
            return qml.prod(ops) if len(ops) > 1 else ops[0]

        except Exception as e:
            logger.error(f"Error creating Pauli product: {str(e)}")
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
            logger.info(f"Created {len(terms)} cost Hamiltonian terms")
            return terms
        except Exception as e:
            logger.error(f"Error in cost_hamiltonian: {str(e)}")
            raise

    def _qaoa_layer(self, gamma: float, beta: float, cost_terms: List[Tuple]):
        """
        Implement single QAOA layer.
        """
        try:
            # Cost unitary
            for coeff, pauli_terms in cost_terms:
                ham = self._create_pauli_product(pauli_terms)
                qml.ApproxTimeEvolution(ham, gamma * coeff, 1)
                logger.debug(f"Applied cost unitary with gamma={str(gamma)}, coeff={coeff}")

            # Mixer unitary
            for i in range(self.n_qubits):
                qml.RX(2 * beta, wires=i)
            logger.debug(f"Applied mixer unitary with beta={str(beta)}")
        except Exception as e:
            logger.error(f"Error in QAOA layer: {str(e)}")
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
            # Initialize parameters with gradient support
            params = qml.numpy.array(np.random.uniform(0, 2*np.pi, 2*self.depth), requires_grad=True)

            def objective(params):
                measurements = self.circuit(params, cost_terms)
                return measurements  # Average over all qubits

            opt = qml.AdamOptimizer(stepsize=0.05) #AdamOptimizer and reduced learning rate
            costs = []

            for step in range(steps):
                params, cost = opt.step_and_cost(objective, params) # step_and_cost
                costs.append(cost)

                if step % 10 == 0:
                    logger.info(f"Step {step}: Cost = {cost:.4f}")

            final_cost = cost
            logger.info(f"Optimization completed. Final cost: {final_cost:.4f}")
            return params, costs

        except Exception as e:
            logger.error(f"Error in optimization: {str(e)}")
            raise

    def prod(self, operators):
        """Helper function to multiply quantum operators"""
        result = operators[0]
        for op in operators[1:]:
            result = result @ op
        return result

def prod(operators):
    """Helper function to multiply quantum operators"""
    result = operators[0]
    for op in operators[1:]:
        result = result @ op
    return result