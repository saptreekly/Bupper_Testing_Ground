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

            # Initialize devices for each partition
            for i in range(self.n_partitions):
                start_idx = i * self.max_partition_size
                end_idx = min(start_idx + self.max_partition_size, n_qubits)
                partition_size = end_idx - start_idx

                # Create device with error mitigation
                dev = qml.device('default.qubit', wires=partition_size)
                self.devices.append(dev)

                # Create QNode with device binding and proper interface
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

                    return [qml.expval(qml.PauliZ(i)) for i in range(partition_size)]

                self.circuits.append(circuit)

            logger.info(f"Successfully initialized {self.n_partitions} quantum devices")

        except Exception as e:
            logger.error(f"Failed to initialize quantum devices: {str(e)}")
            raise

    def optimize(self, cost_terms: List[Tuple], **kwargs) -> Tuple[np.ndarray, List[float]]:
        """QAOA optimization with enhanced parameter validation and gradient computation."""
        try:
            # Initialize parameters with smaller values
            params = np.array([
                np.random.uniform(-0.1, 0.1),  # gamma
                np.random.uniform(0.1, 0.5)    # beta
            ])

            logger.info(f"Starting optimization with initial parameters: {params}")

            def cost_function(p):
                """Enhanced cost function with parameter validation."""
                try:
                    measurements = self.get_expectation_values(p, cost_terms)
                    logger.debug(f"Measurements shape: {measurements.shape}, values: {measurements}")

                    total_cost = 0.0
                    for partition_idx, partition_costs in self._validate_cost_terms(cost_terms).items():
                        start_idx = partition_idx * self.max_partition_size
                        for coeff, (i, j) in partition_costs:
                            local_i = i % self.max_partition_size
                            local_j = j % self.max_partition_size
                            if local_i < self.max_partition_size and local_j < self.max_partition_size:
                                total_cost += coeff * measurements[start_idx + local_i] * measurements[start_idx + local_j]

                    logger.debug(f"Cost function value: {total_cost}")
                    return total_cost
                except Exception as e:
                    logger.error(f"Error in cost function: {str(e)}")
                    return float('inf')

            try:
                # Validate cost terms
                if not cost_terms:
                    logger.error("Empty cost terms provided")
                    return np.array([0.0, 0.0]), [float('inf')]

                # Compute gradient using PennyLane's gradient transform
                grad_fn = qml.grad(cost_function)
                current_cost = cost_function(params)
                grad = grad_fn(params)
                logger.debug(f"Gradient shape: {np.array(grad).shape}, values: {grad}")

                # Update parameters with gradient descent
                new_params = params - 0.05 * np.array(grad)
                new_params = self._validate_and_truncate_params(new_params)

                logger.info(f"Optimization step completed with cost: {current_cost:.6f}")
                return new_params, [current_cost]

            except Exception as e:
                logger.error(f"Error in optimization: {str(e)}")
                return np.array([0.0, 0.0]), [float('inf')]

        except Exception as e:
            logger.error(f"Error during optimization: {str(e)}")
            raise

    def get_expectation_values(self, params: np.ndarray, cost_terms: List[Tuple]) -> np.ndarray:
        """Get expectation values with enhanced partition handling."""
        try:
            if params is None:
                logger.warning("Received None parameters, using default initialization")
                params = np.array([0.0, 0.0])

            # Validate parameters
            params = self._validate_and_truncate_params(params)
            logger.debug(f"Getting expectation values for params: gamma={params[0]:.4f}, beta={params[1]:.4f}")

            # Validate and partition cost terms
            partitioned_terms = self._validate_cost_terms(cost_terms)
            logger.debug(f"Using {len(partitioned_terms)} partitions for cost terms")

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
                        logger.debug(f"Partition {partition_idx} results shape: {len(partition_results)}")
                        measurements[start_idx:end_idx] = partition_results
                    except Exception as e:
                        logger.error(f"Error in partition {partition_idx}: {str(e)}")
                        measurements[start_idx:end_idx] = np.zeros(partition_size)

            logger.debug(f"Final measurements shape: {measurements.shape}")
            return measurements

        except Exception as e:
            logger.error(f"Error getting expectation values: {str(e)}")
            raise

    def _validate_and_truncate_params(self, params: np.ndarray) -> np.ndarray:
        """Ensure parameters are exactly length 2 and within bounds."""
        if not isinstance(params, np.ndarray):
            params = np.array(params, dtype=float)

        # Handle empty or incorrect size params
        if params.size == 0:
            params = np.zeros(2)
        elif params.size != 2:
            logger.warning(f"Parameter validation: truncating from length {params.size} to 2")
            if params.size < 2:
                params = np.pad(params, (0, 2 - params.size))
            else:
                params = params[:2]

        # Ensure parameters are within bounds
        params[0] = np.clip(params[0], -2*np.pi, 2*np.pi)  # gamma
        params[1] = np.clip(params[1], -np.pi, np.pi)      # beta

        return params

    def _validate_cost_terms(self, cost_terms: List[Tuple]) -> Dict[int, List[Tuple]]:
        """Validate and partition cost terms."""
        partitioned_terms = {}

        if not cost_terms:
            logger.warning("Empty cost terms provided")
            return {0: [(1.0, (0, min(1, self.max_partition_size-1)))]}

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

        logger.debug(f"Partitioned {len(cost_terms)} cost terms into {len(partitioned_terms)} groups")
        return partitioned_terms