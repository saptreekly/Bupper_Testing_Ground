import pennylane as qml
import numpy as np
from typing import List, Tuple, Optional, Dict, Callable
import logging
import sys

logger = logging.getLogger(__name__)

class QAOACircuit:
    def __init__(self, n_qubits: int, depth: int = 1, progress_callback: Optional[Callable] = None):
        """Initialize QAOA circuit with enhanced partitioning for large qubit counts."""
        try:
            self.n_qubits = n_qubits
            self.depth = depth
            self.progress_callback = progress_callback

            # Optimize partition size for 5 cities (25 qubits total)
            self.max_partition_size = min(25, n_qubits)  # Adjusted for optimal performance
            self.n_partitions = (n_qubits + self.max_partition_size - 1) // self.max_partition_size

            logger.info(f"Initializing QAOA with {n_qubits} qubits across {self.n_partitions} partitions")
            logger.info(f"Each partition will handle up to {self.max_partition_size} qubits")

            # Enhanced safety checks for large problems
            if n_qubits > 50:
                logger.warning(f"Large problem size detected: {n_qubits} qubits. Consider using hybrid approach.")

            self.devices = []
            self.circuits = []
            self.partition_sizes = []

            for i in range(self.n_partitions):
                start_idx = i * self.max_partition_size
                end_idx = min(start_idx + self.max_partition_size, n_qubits)
                partition_size = end_idx - start_idx
                self.partition_sizes.append(partition_size)

                try:
                    logger.info(f"Creating device for partition {i} with {partition_size} qubits")
                    dev = qml.device('default.qubit', wires=partition_size, shots=None)
                    self.devices.append(dev)

                    @qml.qnode(dev)
                    def circuit(params, cost_terms):
                        # Initial state preparation
                        for i in range(partition_size):
                            qml.Hadamard(wires=i)

                        # Apply QAOA layers
                        gamma, beta = params[0], params[1]

                        # Cost Hamiltonian
                        for coeff, (i, j) in cost_terms:
                            if i != j and i < partition_size and j < partition_size:
                                qml.CNOT(wires=[i, j])
                                qml.RZ(2 * gamma * coeff, wires=j)
                                qml.CNOT(wires=[i, j])

                        # Mixer Hamiltonian
                        for i in range(partition_size):
                            qml.RX(2 * beta, wires=i)

                        # Return expectation values
                        return [qml.expval(qml.PauliZ(i)) for i in range(partition_size)]

                    self.circuits.append(circuit)
                    logger.info(f"Successfully created circuit for partition {i}")

                except Exception as e:
                    logger.error(f"Failed to initialize partition {i}: {str(e)}")
                    raise RuntimeError(f"Device initialization failed for partition {i}: {str(e)}")

        except Exception as e:
            logger.error(f"Failed to initialize quantum devices: {str(e)}")
            raise

    def optimize(self, cost_terms: List[Tuple], **kwargs) -> Tuple[np.ndarray, List[float]]:
        """QAOA optimization with improved parameter handling."""
        try:
            # Initialize parameters
            params = np.array([0.01, np.pi/4])  # Initial guess
            costs = []

            def cost_function(p):
                try:
                    measurements = self.get_expectation_values(p, cost_terms)
                    total_cost = 0.0

                    for partition_idx, partition_costs in self._validate_cost_terms(cost_terms).items():
                        start_idx = partition_idx * self.max_partition_size
                        partition_measurements = measurements[start_idx:start_idx + self.partition_sizes[partition_idx]]

                        for coeff, (i, j) in partition_costs:
                            if i < len(partition_measurements) and j < len(partition_measurements):
                                total_cost += float(coeff) * float(partition_measurements[i]) * float(partition_measurements[j])

                    return float(total_cost)
                except Exception as e:
                    logger.error(f"Error in cost function: {str(e)}")
                    return float('inf')

            # Optimization parameters
            steps = kwargs.get('steps', 100)
            learning_rate = 0.1
            min_learning_rate = 0.01
            decay_rate = 0.995

            current_cost = cost_function(params)
            costs.append(current_cost)

            for step in range(steps):
                try:
                    # Compute gradient
                    grad = np.zeros(2)
                    eps = max(1e-4, learning_rate * 0.1)

                    for i in range(2):
                        params_plus = params.copy()
                        params_plus[i] += eps
                        cost_plus = cost_function(params_plus)
                        grad[i] = (cost_plus - current_cost) / eps

                    # Update parameters
                    grad_norm = np.linalg.norm(grad)
                    if grad_norm > 1.0:
                        grad = grad / grad_norm

                    params -= learning_rate * grad
                    params = self._validate_and_truncate_params(params)

                    # Update learning rate
                    learning_rate = max(min_learning_rate, learning_rate * decay_rate)

                    # Check convergence
                    new_cost = cost_function(params)
                    if len(costs) > 5 and abs(new_cost - costs[-1]) < 1e-6:
                        logger.info(f"Converged at step {step}")
                        break

                    costs.append(new_cost)
                    current_cost = new_cost

                except Exception as e:
                    logger.error(f"Error in optimization step {step}: {str(e)}")
                    continue

            logger.info(f"Final parameters: gamma={params[0]:.4f}, beta={params[1]:.4f}")
            return params, costs

        except Exception as e:
            logger.error(f"Error during optimization: {str(e)}")
            raise

    def get_expectation_values(self, params: np.ndarray, cost_terms: List[Tuple]) -> np.ndarray:
        """Compute expectation values with enhanced error handling."""
        try:
            params = self._validate_and_truncate_params(params)
            measurements = np.zeros(self.n_qubits)

            # Process each partition
            for partition_idx, partition_costs in self._validate_cost_terms(cost_terms).items():
                start_idx = partition_idx * self.max_partition_size
                partition_size = self.partition_sizes[partition_idx]

                try:
                    circuit = self.circuits[partition_idx]
                    partition_results = circuit(params, partition_costs)
                    measurements[start_idx:start_idx + partition_size] = partition_results

                except Exception as e:
                    logger.error(f"Error in partition {partition_idx}: {str(e)}")
                    measurements[start_idx:start_idx + partition_size] = np.zeros(partition_size)

            return measurements

        except Exception as e:
            logger.error(f"Error getting expectation values: {str(e)}")
            raise

    def _validate_and_truncate_params(self, params: np.ndarray) -> np.ndarray:
        """Parameter validation with improved bounds."""
        if not isinstance(params, np.ndarray):
            params = np.array(params, dtype=float)

        if params.size == 0:
            params = np.array([0.01, np.pi/4])
        elif params.size != 2:
            if params.size < 2:
                params = np.pad(params, (0, 2 - params.size))
            else:
                params = params[:2]

        # Parameter bounds
        params[0] = np.clip(params[0], -2*np.pi, 2*np.pi)  # gamma
        params[1] = np.clip(params[1], -np.pi, np.pi)      # beta

        return params

    def _validate_cost_terms(self, cost_terms: List[Tuple]) -> Dict[int, List[Tuple]]:
        """Enhanced cost term validation with improved handling of large systems."""
        partitioned_terms = {i: [] for i in range(self.n_partitions)}

        if not cost_terms:
            logger.warning("Empty cost terms provided")
            return {0: [(1.0, (0, 1))]}

        # Normalize coefficients
        coeffs = [abs(coeff) for coeff, _ in cost_terms]
        max_coeff = max(coeffs) if coeffs else 1.0

        # Process and partition terms
        for coeff, (i, j) in cost_terms:
            if i == j:  # Skip self-loops
                continue

            partition_i = i // self.max_partition_size
            partition_j = j // self.max_partition_size

            if partition_i == partition_j:
                local_i = i % self.max_partition_size
                local_j = j % self.max_partition_size

                if (local_i < self.partition_sizes[partition_i] and
                    local_j < self.partition_sizes[partition_i]):
                    norm_coeff = coeff / max_coeff if max_coeff > 0 else coeff
                    partitioned_terms[partition_i].append((norm_coeff, (local_i, local_j)))

        logger.debug(f"Partitioned {len(cost_terms)} cost terms into {len(partitioned_terms)} groups")
        return partitioned_terms