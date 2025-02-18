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

            # Adjust partition size based on total qubits
            self.max_partition_size = min(25, n_qubits)  # Maximum supported by PennyLane
            self.n_partitions = (n_qubits + self.max_partition_size - 1) // self.max_partition_size

            logger.info(f"Initializing QAOA with {n_qubits} qubits across {self.n_partitions} partitions")
            logger.info(f"Each partition will handle up to {self.max_partition_size} qubits")

            if n_qubits > 100:  # Safety check for very large problems
                logger.warning(f"Large problem size detected: {n_qubits} qubits. This may impact performance.")

            self.devices = []
            self.circuits = []
            self.partition_sizes = []  # Store actual sizes of each partition

            for i in range(self.n_partitions):
                start_idx = i * self.max_partition_size
                end_idx = min(start_idx + self.max_partition_size, n_qubits)
                partition_size = end_idx - start_idx
                self.partition_sizes.append(partition_size)

                try:
                    logger.info(f"Creating device for partition {i} with {partition_size} qubits")
                    device_params = {
                        'name': 'default.qubit',
                        'wires': partition_size,
                        'shots': None
                    }
                    logger.debug(f"Device parameters for partition {i}: {device_params}")

                    dev = qml.device(**device_params)
                    logger.info(f"Successfully created device for partition {i}")
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
                            if i < partition_size and j < partition_size:
                                qml.CNOT(wires=[i, j])
                                qml.RZ(2 * gamma * coeff, wires=j)
                                qml.CNOT(wires=[i, j])

                        # Mixer Hamiltonian
                        for i in range(partition_size):
                            qml.RX(2 * beta, wires=i)

                        return [qml.expval(qml.PauliZ(i)) for i in range(partition_size)]

                    self.circuits.append(circuit)

                    if self.progress_callback:
                        progress = 0.1 + (0.4 * (i + 1) / self.n_partitions)
                        self.progress_callback(0, {
                            "status": f"Initializing partition {i+1}/{self.n_partitions}",
                            "progress": progress
                        })

                except Exception as e:
                    logger.error(f"Failed to initialize partition {i}: {str(e)}", exc_info=True)
                    raise RuntimeError(f"Device initialization failed for partition {i}: {str(e)}")

            logger.info(f"Successfully initialized {self.n_partitions} quantum devices")

        except Exception as e:
            logger.error(f"Failed to initialize quantum devices: {str(e)}", exc_info=True)
            raise

    def optimize(self, cost_terms: List[Tuple], **kwargs) -> Tuple[np.ndarray, List[float]]:
        """QAOA optimization with enhanced memory management."""
        try:
            params = np.array([0.01, 0.1])  # Reduced initial values for better convergence
            logger.info(f"Starting optimization with initial parameters: {params}")

            def cost_function(p):
                try:
                    measurements = self.get_expectation_values(p, cost_terms)
                    total_cost = 0.0
                    for partition_idx, partition_costs in self._validate_cost_terms(cost_terms).items():
                        start_idx = partition_idx * self.max_partition_size
                        partition_measurements = measurements[start_idx:start_idx + self.partition_sizes[partition_idx]]

                        for coeff, (i, j) in partition_costs:
                            if i < len(partition_measurements) and j < len(partition_measurements):
                                total_cost += coeff * partition_measurements[i] * partition_measurements[j]
                    logger.debug(f"Cost function evaluation: params={p}, cost={total_cost}")
                    return float(total_cost)  # Ensure we return a scalar
                except Exception as e:
                    logger.error(f"Error in cost function: {str(e)}")
                    return float('inf')

            try:
                # Evaluate cost before gradient computation
                current_cost = cost_function(params)
                logger.info(f"Initial cost: {current_cost}")

                # Compute gradient with error handling
                try:
                    grad = qml.grad(cost_function)(params)
                    if not isinstance(grad, np.ndarray) or grad.size != 2:
                        logger.error(f"Invalid gradient shape: {grad.shape if hasattr(grad, 'shape') else 'not array'}")
                        grad = np.zeros(2)
                except Exception as e:
                    logger.error(f"Gradient computation failed: {str(e)}", exc_info=True)
                    grad = np.zeros(2)

                # Update parameters with gradient clipping
                grad_norm = np.linalg.norm(grad)
                if grad_norm > 1.0:
                    grad = grad / grad_norm

                logger.info(f"Gradient: {grad}, Norm: {grad_norm}")
                new_params = params - 0.01 * grad
                new_params = self._validate_and_truncate_params(new_params)

                if self.progress_callback:
                    self.progress_callback(1, {
                        "status": "Optimization step completed",
                        "progress": 0.8,
                        "cost": current_cost
                    })

                logger.info(f"Optimization step completed with cost: {current_cost:.6f}")
                return new_params, [current_cost]

            except Exception as e:
                logger.error(f"Error in optimization: {str(e)}", exc_info=True)
                return np.array([0.0, 0.0]), [float('inf')]

        except Exception as e:
            logger.error(f"Error during optimization: {str(e)}")
            raise

    def get_expectation_values(self, params: np.ndarray, cost_terms: List[Tuple]) -> np.ndarray:
        """Memory-efficient expectation value computation."""
        try:
            if params is None:
                logger.warning("Received None parameters, using default initialization")
                params = np.array([0.0, 0.0])

            params = self._validate_and_truncate_params(params)
            logger.info(f"Computing expectation values with gamma={params[0]:.4f}, beta={params[1]:.4f}")

            partitioned_terms = self._validate_cost_terms(cost_terms)
            measurements = np.zeros(self.n_qubits)

            for partition_idx, partition_costs in partitioned_terms.items():
                start_idx = partition_idx * self.max_partition_size
                partition_size = self.partition_sizes[partition_idx]

                try:
                    circuit = self.circuits[partition_idx]
                    partition_results = circuit(params, partition_costs)
                    measurements[start_idx:start_idx + partition_size] = partition_results

                    if self.progress_callback:
                        progress = 0.6 + (0.2 * (partition_idx + 1) / len(partitioned_terms))
                        self.progress_callback(partition_idx, {
                            "status": f"Computing partition {partition_idx + 1}/{len(partitioned_terms)}",
                            "progress": progress
                        })

                except Exception as e:
                    logger.error(f"Error in partition {partition_idx}: {str(e)}", exc_info=True)
                    measurements[start_idx:start_idx + partition_size] = np.zeros(partition_size)

            return measurements

        except Exception as e:
            logger.error(f"Error getting expectation values: {str(e)}", exc_info=True)
            raise

    def _validate_and_truncate_params(self, params: np.ndarray) -> np.ndarray:
        """Parameter validation with improved bounds checking."""
        if not isinstance(params, np.ndarray):
            params = np.array(params, dtype=float)

        if params.size == 0:
            params = np.zeros(2)
        elif params.size != 2:
            logger.warning(f"Parameter validation: truncating from length {params.size} to 2")
            if params.size < 2:
                params = np.pad(params, (0, 2 - params.size))
            else:
                params = params[:2]

        params[0] = np.clip(params[0], -np.pi, np.pi)      # gamma
        params[1] = np.clip(params[1], -np.pi/2, np.pi/2)  # beta

        return params

    def _validate_cost_terms(self, cost_terms: List[Tuple]) -> Dict[int, List[Tuple]]:
        """Improved cost term validation and partitioning."""
        partitioned_terms = {}

        if not cost_terms:
            logger.warning("Empty cost terms provided")
            return {0: [(1.0, (0, 1))]}

        # Initialize all partitions
        for i in range(self.n_partitions):
            partitioned_terms[i] = []

        coeffs = [abs(coeff) for coeff, _ in cost_terms]
        max_coeff = max(coeffs) if coeffs else 1.0

        for coeff, (i, j) in cost_terms:
            partition_i = i // self.max_partition_size
            partition_j = j // self.max_partition_size

            if partition_i == partition_j:  # Only handle intra-partition terms
                if partition_i not in partitioned_terms:
                    partitioned_terms[partition_i] = []

                local_i = i % self.max_partition_size
                local_j = j % self.max_partition_size

                # Check if indices are within partition size
                if local_i < self.partition_sizes[partition_i] and local_j < self.partition_sizes[partition_i]:
                    norm_coeff = coeff / max_coeff if max_coeff > 0 else coeff
                    partitioned_terms[partition_i].append((norm_coeff, (local_i, local_j)))

        logger.debug(f"Partitioned {len(cost_terms)} cost terms into {len(partitioned_terms)} groups")
        return partitioned_terms