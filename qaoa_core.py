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

            # Increase max partition size for 5 cities
            self.max_partition_size = min(30, n_qubits)  # Increased from 25 to handle 5 cities
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

                        # Cost Hamiltonian with improved efficiency
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
        """Enhanced QAOA optimization with improved convergence for larger systems."""
        try:
            # Initialize parameters with improved scaling
            params = np.array([0.01, np.pi/4])  # Better initial guess for mixer
            costs = []

            logger.info(f"Starting QAOA optimization with {len(cost_terms)} cost terms")
            logger.info(f"Initial parameters: gamma={params[0]:.4f}, beta={params[1]:.4f}")

            def cost_function(p):
                try:
                    measurements = self.get_expectation_values(p, cost_terms)
                    total_cost = 0.0

                    # Process partitions with enhanced error checking
                    for partition_idx, partition_costs in self._validate_cost_terms(cost_terms).items():
                        start_idx = partition_idx * self.max_partition_size
                        partition_measurements = measurements[start_idx:start_idx + self.partition_sizes[partition_idx]]

                        partition_cost = 0.0
                        for coeff, (i, j) in partition_costs:
                            if i < len(partition_measurements) and j < len(partition_measurements):
                                term_cost = float(coeff) * float(partition_measurements[i]) * float(partition_measurements[j])
                                partition_cost += term_cost

                        logger.debug(f"Partition {partition_idx} cost: {partition_cost:.6f}")
                        total_cost += partition_cost

                    logger.debug(f"Total cost for parameters gamma={p[0]:.4f}, beta={p[1]:.4f}: {total_cost:.6f}")
                    return float(total_cost)
                except Exception as e:
                    logger.error(f"Error in cost function: {str(e)}")
                    return float('inf')

            # Enhanced optimization with adaptive learning rate
            steps = kwargs.get('steps', 100)
            learning_rate = 0.1
            min_learning_rate = 0.01
            decay_rate = 0.995

            current_cost = cost_function(params)
            costs.append(current_cost)
            logger.info(f"Initial cost: {current_cost:.6f}")

            best_cost = current_cost
            best_params = params.copy()
            no_improvement_count = 0

            for step in range(steps):
                try:
                    # Compute gradient with improved numerical stability
                    grad = np.zeros(2)
                    eps = max(1e-4, learning_rate * 0.1)

                    for i in range(2):
                        params_plus = params.copy()
                        params_plus[i] += eps
                        cost_plus = cost_function(params_plus)
                        grad[i] = (cost_plus - current_cost) / eps
                        logger.debug(f"Gradient {i}: {grad[i]:.6f}")

                    # Update parameters with gradient clipping
                    grad_norm = np.linalg.norm(grad)
                    if grad_norm > 1.0:
                        grad = grad / grad_norm
                        logger.debug(f"Clipped gradient norm from {grad_norm:.6f} to 1.0")

                    # Apply update with adaptive learning rate
                    params -= learning_rate * grad
                    params = self._validate_and_truncate_params(params)

                    # Update learning rate
                    learning_rate = max(min_learning_rate, learning_rate * decay_rate)
                    logger.debug(f"Updated learning rate: {learning_rate:.6f}")

                    # Check convergence
                    new_cost = cost_function(params)
                    improvement = (current_cost - new_cost) / abs(current_cost) if abs(current_cost) > 1e-10 else 0.0

                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_params = params.copy()
                        no_improvement_count = 0
                        logger.info(f"Step {step}: New best cost = {best_cost:.6f}")
                    else:
                        no_improvement_count += 1

                    logger.info(f"Step {step}: Cost = {new_cost:.6f}, Improvement = {improvement:.6f}, No improvement count = {no_improvement_count}")

                    costs.append(new_cost)
                    current_cost = new_cost

                    if no_improvement_count >= 10:
                        logger.info(f"Early stopping triggered after {step} steps due to no improvement")
                        break

                except Exception as e:
                    logger.error(f"Error in optimization step {step}: {str(e)}")
                    continue

            logger.info(f"Optimization completed: Best cost = {best_cost:.6f}")
            logger.info(f"Final parameters: gamma={best_params[0]:.4f}, beta={best_params[1]:.4f}")
            return best_params, costs

        except Exception as e:
            logger.error(f"Error during optimization: {str(e)}")
            raise

    def get_expectation_values(self, params: np.ndarray, cost_terms: List[Tuple]) -> np.ndarray:
        """Compute expectation values with enhanced error handling for large systems."""
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
        """Parameter validation with improved bounds for 5-city problems."""
        if not isinstance(params, np.ndarray):
            params = np.array(params, dtype=float)

        if params.size == 0:
            params = np.array([0.01, np.pi/4])  # Better initialization
        elif params.size != 2:
            if params.size < 2:
                params = np.pad(params, (0, 2 - params.size))
            else:
                params = params[:2]

        # Expanded bounds for better exploration
        params[0] = np.clip(params[0], -2*np.pi, 2*np.pi)  # gamma
        params[1] = np.clip(params[1], -np.pi, np.pi)      # beta

        return params

    def _validate_cost_terms(self, cost_terms: List[Tuple]) -> Dict[int, List[Tuple]]:
        """Enhanced cost term validation with improved handling of large systems."""
        partitioned_terms = {i: [] for i in range(self.n_partitions)}

        if not cost_terms:
            logger.warning("Empty cost terms provided")
            return {0: [(1.0, (0, 1))]}

        # Normalize coefficients with improved numerical stability
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