import pennylane as qml
import numpy as np
from typing import List, Tuple
import qiskit
from pennylane import qnode
import logging

logger = logging.getLogger(__name__)

class QAOACircuit:
    def __init__(self, n_qubits: int, depth: int = 1):
        """
        Initialize QAOA circuit for routing optimization.

        Args:
            n_qubits (int): Number of qubits needed for problem encoding
            depth (int): Number of QAOA layers (p-value)
        """
        self.n_qubits = n_qubits
        self.depth = depth
        try:
            # Create the quantum device
            self.dev = qml.device('default.qubit', wires=n_qubits, shots=1000)
            # Define the circuit as a QNode
            self.circuit = qnode(self.dev)(self._circuit_implementation)
            logger.info(f"Successfully initialized quantum device with {n_qubits} qubits")
        except Exception as e:
            logger.error(f"Failed to initialize quantum device: {str(e)}")
            raise

    def _circuit_implementation(self, params, cost_terms):
        """
        Implementation of the QAOA circuit.

        Args:
            params (np.ndarray): Circuit parameters [gamma1, beta1, gamma2, beta2, ...]
            cost_terms (List[Tuple]): Cost Hamiltonian terms

        Returns:
            List[float]: Cost expectation value
        """
        try:
            # Initialize in superposition
            for i in range(self.n_qubits):
                qml.Hadamard(wires=i)

            # Apply QAOA layers
            for layer in range(self.depth):
                gamma = params[2 * layer]
                beta = params[2 * layer + 1]
                self._qaoa_layer(gamma, beta, cost_terms)

            # Create cost Hamiltonian operator
            cost_op = sum(coeff * self._create_pauli_product(pauli_terms)
                         for coeff, pauli_terms in cost_terms)

            # Return expectation value as a measurement observable
            measurement = qml.expval(cost_op)
            logger.debug(f"Raw measurement type: {type(measurement)}")
            logger.debug(f"Raw measurement value: {measurement}")

            # Return as list to maintain PennyLane's measurement format
            return [measurement]

        except Exception as e:
            logger.error(f"Error in circuit execution: {str(e)}")
            raise

    def _create_pauli_product(self, pauli_terms):
        """Helper function to create a product of Pauli operators"""
        try:
            operators = [qml.PauliZ(int(term[1:])) for term in pauli_terms]
            if not operators:
                return qml.Identity(0)
            result = operators[0]
            for op in operators[1:]:
                result = result @ op
            return result
        except Exception as e:
            logger.error(f"Error creating Pauli product: {str(e)}")
            raise

    def _qaoa_layer(self, gamma: float, beta: float, cost_terms: List[Tuple]):
        """
        Implement single QAOA layer.

        Args:
            gamma (float): Parameter for cost unitary
            beta (float): Parameter for mixer unitary
            cost_terms (List[Tuple]): Cost Hamiltonian terms
        """
        try:
            # Cost unitary
            for coeff, pauli_terms in cost_terms:
                ham = self._create_pauli_product(pauli_terms)
                qml.ApproxTimeEvolution(ham, gamma * coeff, 1)

            # Mixer unitary
            for i in range(self.n_qubits):
                qml.RX(2 * beta, wires=i)
        except Exception as e:
            logger.error(f"Error in QAOA layer: {str(e)}")
            raise

    def cost_hamiltonian(self, adjacency_matrix: np.ndarray) -> List[Tuple]:
        """
        Construct cost Hamiltonian terms from adjacency matrix.

        Args:
            adjacency_matrix (np.ndarray): Matrix representing distances between nodes

        Returns:
            List[Tuple]: List of (coefficient, Pauli terms) pairs
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

    def mixer_hamiltonian(self) -> List[Tuple]:
        """
        Construct mixer Hamiltonian terms.

        Returns:
            List[Tuple]: List of (coefficient, Pauli terms) pairs
        """
        return [(1.0, [f"X{i}"]) for i in range(self.n_qubits)]

    def optimize(self, cost_terms: List[Tuple], steps: int = 100):
        """
        Optimize QAOA parameters.

        Args:
            cost_terms (List[Tuple]): Cost Hamiltonian terms
            steps (int): Number of optimization steps

        Returns:
            Tuple[np.ndarray, float]: Optimal parameters and final cost
        """
        try:
            # Initialize parameters with gradient support
            params = qml.numpy.array(np.random.uniform(0, 2*np.pi, 2*self.depth), requires_grad=True)

            def objective(params):
                measurements = self.circuit(params, cost_terms)
                return measurements  # Average over all qubits

            opt = qml.GradientDescentOptimizer(stepsize=0.1)
            costs = []

            for step in range(steps):
                params = opt.step(objective, params)
                current_cost = float(objective(params))
                costs.append(current_cost)

                if step % 10 == 0:
                    logger.info(f"Step {step}: Cost = {current_cost:.4f}")

            final_cost = float(objective(params))
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