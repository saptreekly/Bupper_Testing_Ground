import pennylane as qml
import numpy as np
from typing import List, Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)

class QAOACircuit:
    def __init__(self, n_qubits: int, depth: int = 1):
        """Initialize QAOA circuit with enhanced partitioning for large qubit counts."""
        try:
            # Maximum partition size for PennyLane with safety margin
            self.max_partition_size = 25
            self.n_partitions = (n_qubits + self.max_partition_size - 1) // self.max_partition_size

            logger.info(f"Initializing QAOA with {n_qubits} qubits across {self.n_partitions} partitions")

            self.n_qubits = n_qubits
            self.depth = 1  # Force depth to 1 to maintain 2-parameter system
            self.devices = []
            self.circuits = []

            # Initialize parameters array with validation
            self.params = np.array([0.0, 0.0])  # [gamma, beta]

            # Initialize devices for each partition
            for i in range(self.n_partitions):
                start_idx = i * self.max_partition_size
                end_idx = min(start_idx + self.max_partition_size, n_qubits)
                partition_size = end_idx - start_idx

                # Create device with error mitigation
                dev = qml.device('default.qubit', wires=partition_size, shots=None)
                self.devices.append(dev)

                # Create QNode with device binding
                @qml.qnode(dev, interface="autograd", diff_method="parameter-shift")
                def circuit(params, cost_terms):
                    # Initial state preparation
                    for i in range(partition_size):
                        qml.Hadamard(wires=i)

                    gamma, beta = params

                    # Cost Hamiltonian
                    for coeff, (i, j) in cost_terms:
                        if i != j and i < partition_size and j < partition_size:
                            qml.CNOT(wires=[i, j])
                            qml.RZ(2 * gamma * coeff, wires=j)
                            qml.CNOT(wires=[i, j])
                        elif i < partition_size:
                            qml.RZ(2 * gamma * coeff, wires=i)

                    # Mixer Hamiltonian
                    for i in range(partition_size):
                        qml.RX(2 * beta, wires=i)

                    # Return expectation values for all qubits
                    return [qml.expval(qml.PauliZ(i)) for i in range(partition_size)]

                self.circuits.append(circuit)

            logger.info(f"Successfully initialized {self.n_partitions} quantum devices")

        except Exception as e:
            logger.error(f"Failed to initialize quantum devices: {str(e)}")
            raise

    def get_expectation_values(self, params: np.ndarray, cost_terms: List[Tuple]) -> np.ndarray:
        """Get expectation values with enhanced partition handling."""
        try:
            # Validate parameters
            params = self._validate_and_truncate_params(params)
            logger.info(f"Computing expectation values with gamma={params[0]:.4f}, beta={params[1]:.4f}")

            # Validate and partition cost terms
            partitioned_terms = self._validate_cost_terms(cost_terms)
            measurements = np.zeros(self.n_qubits)

            # Execute circuits for each partition
            for partition_idx, partition_costs in partitioned_terms.items():
                start_idx = partition_idx * self.max_partition_size
                end_idx = min(start_idx + self.max_partition_size, self.n_qubits)
                partition_size = end_idx - start_idx

                if partition_size > 0 and partition_idx < len(self.circuits):
                    try:
                        circuit = self.circuits[partition_idx]
                        partition_results = circuit(params, partition_costs)
                        measurements[start_idx:end_idx] = partition_results[:partition_size]
                    except Exception as e:
                        logger.error(f"Error in partition {partition_idx}: {str(e)}")
                        measurements[start_idx:end_idx] = np.zeros(partition_size)

            return measurements

        except Exception as e:
            logger.error(f"Error getting expectation values: {str(e)}")
            raise

    def _validate_and_truncate_params(self, params: np.ndarray) -> np.ndarray:
        """Ensure parameters are exactly length 2 and within bounds."""
        if not isinstance(params, np.ndarray):
            params = np.array(params)

        # Handle empty or incorrect size params
        if len(params) == 0:
            params = np.zeros(2)
        elif len(params) != 2:
            logger.warning(f"Parameter validation: truncating from length {len(params)} to 2")
            if len(params) < 2:
                params = np.pad(params, (0, 2 - len(params)))
            else:
                params = params[:2]

        # Ensure parameters are within bounds
        params[0] = np.clip(params[0], -2*np.pi, 2*np.pi)  # gamma
        params[1] = np.clip(params[1], -np.pi, np.pi)      # beta

        return params

    def _validate_cost_terms(self, cost_terms: List[Tuple]) -> Dict[int, List[Tuple]]:
        """Validate and partition cost terms."""
        if not cost_terms:
            logger.warning("Empty cost terms provided")
            return {0: [(1.0, (0, min(1, self.max_partition_size-1)))]}

        partitioned_terms = {}
        coeffs = [abs(coeff) for coeff, _ in cost_terms]
        max_coeff = max(coeffs) if coeffs else 1.0

        # Initialize all partitions
        for i in range(self.n_partitions):
            partitioned_terms[i] = []

        # Process and assign terms to partitions
        for coeff, (i, j) in cost_terms:
            partition_i = i // self.max_partition_size
            partition_j = j // self.max_partition_size

            if partition_i == partition_j:  # Only handle intra-partition terms
                if partition_i not in partitioned_terms:
                    partitioned_terms[partition_i] = []

                # Adjust indices for partition
                local_i = i % self.max_partition_size
                local_j = j % self.max_partition_size

                norm_coeff = coeff / max_coeff if max_coeff > 0 else coeff
                partitioned_terms[partition_i].append((norm_coeff, (local_i, local_j)))

        # Ensure each partition has at least one term
        for i in range(self.n_partitions):
            if not partitioned_terms[i]:
                start_idx = i * self.max_partition_size
                end_idx = min(start_idx + self.max_partition_size, self.n_qubits)
                if end_idx - start_idx > 1:
                    partitioned_terms[i] = [(1.0, (0, 1))]

        logger.info(f"Partitioned cost terms into {len(partitioned_terms)} groups")
        return partitioned_terms

    def optimize(self, cost_terms: List[Tuple], steps: int = 100) -> Tuple[np.ndarray, List[float]]:
        """QAOA optimization with enhanced parameter validation."""
        try:
            # Initialize parameters
            params = np.array([
                np.random.uniform(-0.1, 0.1),        # gamma
                np.random.uniform(np.pi/4, np.pi/2)  # beta
            ])

            logger.info(f"Starting optimization with initial parameters: {params}")

            opt = qml.AdamOptimizer(stepsize=0.05)
            costs = []
            best_params = None
            best_cost = float('inf')
            no_improvement_count = 0
            min_improvement = 1e-4

            def cost_function(p):
                """Enhanced cost function with parameter validation."""
                try:
                    p = self._validate_and_truncate_params(p)
                    measurements = self.get_expectation_values(p, cost_terms)
                    total_cost = 0.0

                    for partition_idx, partition_costs in self._validate_cost_terms(cost_terms).items():
                        start_idx = partition_idx * self.max_partition_size
                        for coeff, (i, j) in partition_costs:
                            local_i = start_idx + i
                            local_j = start_idx + j
                            if local_i < self.n_qubits and local_j < self.n_qubits:
                                total_cost += coeff * measurements[local_i] * measurements[local_j]
                    return float(total_cost)
                except Exception as e:
                    logger.error(f"Error in cost function: {str(e)}")
                    return float('inf')

            # Optimization loop with enhanced validation
            for step in range(steps):
                try:
                    params = self._validate_and_truncate_params(params)
                    current_cost = cost_function(params)
                    costs.append(float(current_cost))

                    if current_cost < best_cost - min_improvement:
                        best_cost = current_cost
                        best_params = params.copy()
                        no_improvement_count = 0
                        logger.info(f"Step {step}: New best cost = {current_cost:.6f}")
                    else:
                        no_improvement_count += 1

                    if no_improvement_count >= 10:
                        logger.info(f"Early stopping triggered at step {step}")
                        break

                    # Update parameters with validation
                    params = self._validate_and_truncate_params(
                        opt.apply_grad(cost_function, params)
                    )

                except Exception as e:
                    logger.error(f"Error in optimization step {step}: {str(e)}")
                    continue

            final_params = self._validate_and_truncate_params(
                best_params if best_params is not None else params
            )
            logger.info(f"Optimization complete with parameters: gamma={final_params[0]:.4f}, beta={final_params[1]:.4f}")

            return final_params, costs

        except Exception as e:
            logger.error(f"Error during optimization: {str(e)}")
            raise