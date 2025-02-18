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
            self.max_partition_size = min(n_qubits, 20)  # Dynamically adjust partition size
            self.n_qubits = n_qubits
            self.depth = depth
            self.progress_callback = progress_callback

            # Calculate optimal number of partitions
            self.n_partitions = (n_qubits + self.max_partition_size - 1) // self.max_partition_size
            total_qubits = n_qubits * n_qubits  # For routing problems

            logger.info(f"Initializing QAOA with {total_qubits} total qubits across {self.n_partitions} partitions")
            logger.info(f"Each partition will handle up to {self.max_partition_size} qubits")

            if total_qubits > 100:  # Safety check for very large problems
                logger.warning(f"Large problem size detected: {total_qubits} qubits. This may impact performance.")

            self.devices = []
            self.circuits = []

            if self.progress_callback:
                self.progress_callback(0, {"status": "Initializing quantum devices", "progress": 0.1})

            # Initialize devices for each partition
            for i in range(self.n_partitions):
                start_idx = i * self.max_partition_size
                end_idx = min(start_idx + self.max_partition_size, n_qubits)
                partition_size = end_idx - start_idx

                try:
                    # Create device with optimized settings for statevector simulation
                    dev = qml.device('default.qubit', wires=partition_size)
                    self.devices.append(dev)

                    # Create QNode with simplified measurement logic
                    @qml.qnode(dev)
                    def circuit(params, cost_terms):
                        # Prepare initial state
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

                        # Return measurements directly without additional wrapping
                        return [qml.expval(qml.PauliZ(i)) for i in range(partition_size)]

                    self.circuits.append(circuit)
                    logger.debug(f"Successfully initialized partition {i} with {partition_size} qubits")

                    if self.progress_callback:
                        progress = 0.1 + (0.4 * (i + 1) / self.n_partitions)
                        self.progress_callback(0, {
                            "status": f"Initializing partition {i+1}/{self.n_partitions}",
                            "progress": progress
                        })

                except Exception as e:
                    logger.error(f"Failed to initialize partition {i}: {str(e)}")
                    raise

            logger.info(f"Successfully initialized {self.n_partitions} quantum devices")

        except Exception as e:
            logger.error(f"Failed to initialize quantum devices: {str(e)}")
            raise

    def optimize(self, cost_terms: List[Tuple], **kwargs) -> Tuple[np.ndarray, List[float]]:
        """QAOA optimization with enhanced memory management."""
        try:
            # Initialize parameters with smaller values for stability
            params = np.array([0.01, 0.1])  # Reduced initial values for better convergence
            logger.info(f"Starting optimization with initial parameters: {params}")

            def cost_function(p):
                """Memory-efficient cost function implementation."""
                try:
                    measurements = self.get_expectation_values(p, cost_terms)
                    total_cost = 0.0
                    for partition_idx, partition_costs in self._validate_cost_terms(cost_terms).items():
                        start_idx = partition_idx * self.max_partition_size
                        for coeff, (i, j) in partition_costs:
                            local_i = i % self.max_partition_size
                            local_j = j % self.max_partition_size
                            if local_i < self.max_partition_size and local_j < self.max_partition_size:
                                total_cost += coeff * measurements[start_idx + local_i] * measurements[start_idx + local_j]
                    return total_cost
                except Exception as e:
                    logger.error(f"Error in cost function: {str(e)}")
                    return float('inf')

            try:
                # Compute gradient and update parameters
                grad_fn = qml.grad(cost_function)
                current_cost = cost_function(params)
                grad = grad_fn(params)

                # Update parameters with reduced learning rate
                new_params = params - 0.01 * np.array(grad)
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
                logger.error(f"Error in optimization: {str(e)}")
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
            logger.debug(f"Using {len(partitioned_terms)} partitions for cost terms")

            measurements = np.zeros(self.n_qubits)

            for partition_idx, partition_costs in partitioned_terms.items():
                start_idx = partition_idx * self.max_partition_size
                end_idx = min(start_idx + self.max_partition_size, self.n_qubits)
                partition_size = end_idx - start_idx

                if partition_size > 0 and partition_idx < len(self.circuits):
                    try:
                        circuit = self.circuits[partition_idx]
                        # Execute circuit with direct measurement output
                        partition_results = circuit(params, partition_costs)
                        # Store results without additional array operations
                        measurements[start_idx:end_idx] = partition_results

                        if self.progress_callback:
                            progress = 0.6 + (0.2 * (partition_idx + 1) / len(partitioned_terms))
                            self.progress_callback(partition_idx, {
                                "status": f"Computing partition {partition_idx + 1}/{len(partitioned_terms)}",
                                "progress": progress,
                                "measurements": measurements.tolist()
                            })

                    except Exception as e:
                        logger.error(f"Error in partition {partition_idx}: {str(e)}")
                        measurements[start_idx:end_idx] = np.zeros(partition_size)

            logger.debug(f"Final measurements shape: {measurements.shape}")
            return measurements

        except Exception as e:
            logger.error(f"Error getting expectation values: {str(e)}")
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

        # Tighter parameter bounds for stability
        params[0] = np.clip(params[0], -np.pi, np.pi)      # gamma
        params[1] = np.clip(params[1], -np.pi/2, np.pi/2)  # beta

        return params

    def _validate_cost_terms(self, cost_terms: List[Tuple]) -> Dict[int, List[Tuple]]:
        """Improved cost term validation and partitioning."""
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

                local_i = i % self.max_partition_size
                local_j = j % self.max_partition_size

                # Normalize coefficients for numerical stability
                norm_coeff = coeff / max_coeff if max_coeff > 0 else coeff
                partitioned_terms[partition_i].append((norm_coeff, (local_i, local_j)))

        logger.debug(f"Partitioned {len(cost_terms)} cost terms into {len(partitioned_terms)} groups")
        return partitioned_terms