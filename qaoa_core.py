import pennylane as qml
import numpy as np
from typing import List, Tuple
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

                dev = qml.device('default.qubit', wires=partition_size)
                self.devices.append(dev)

                # Create QNode with device binding
                def circuit_closure(params, partition_costs, device=dev):
                    return self._circuit_implementation(params, partition_costs, device)

                qnode = qml.QNode(circuit_closure, 
                                device,
                                interface="autograd",
                                diff_method="backprop")
                self.circuits.append(qnode)

            logger.info(f"Successfully initialized {self.n_partitions} quantum devices")

        except Exception as e:
            logger.error("Failed to initialize quantum devices: %s", str(e))
            raise

    def _circuit_implementation(self, params, partition_costs, device):
        """QAOA circuit implementation for a single partition."""
        try:
            # Explicitly validate parameters
            params = self._validate_and_truncate_params(params)
            gamma, beta = params

            partition_size = len(device.wires)
            logger.debug(f"Circuit implementation for partition size {partition_size}")

            # Initial state preparation
            for i in range(partition_size):
                qml.Hadamard(wires=i)

            # Cost Hamiltonian phase
            for coeff, (i, j) in partition_costs:
                if i != j and i < partition_size and j < partition_size:
                    qml.CNOT(wires=[i, j])
                    qml.RZ(2 * gamma * coeff, wires=j)
                    qml.CNOT(wires=[i, j])
                elif i < partition_size:
                    qml.RZ(2 * gamma * coeff, wires=i)

            # Mixer Hamiltonian phase
            for i in range(partition_size):
                qml.RX(2 * beta, wires=i)

            # Return expectation values
            return [qml.expval(qml.PauliZ(i)) for i in range(partition_size)]

        except Exception as e:
            logger.error("Error in circuit implementation: %s", str(e))
            raise

    def _validate_and_truncate_params(self, params):
        """Ensure parameters are exactly length 2 and within bounds."""
        if not isinstance(params, np.ndarray):
            params = np.array(params)

        if len(params) != 2:
            logger.warning(f"Parameter validation: truncating from length {len(params)} to 2")
            params = params[:2]

        # Ensure parameters are within bounds
        params[0] = np.clip(params[0], -2*np.pi, 2*np.pi)  # gamma
        params[1] = np.clip(params[1], -np.pi, np.pi)      # beta

        return params

    def get_expectation_values(self, params, cost_terms):
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
            logger.error("Error getting expectation values: %s", str(e))
            raise

    def _validate_cost_terms(self, cost_terms: List[Tuple]) -> dict:
        """Validate and partition cost terms."""
        if not cost_terms:
            logger.warning("Empty cost terms provided")
            return {0: [(1.0, (0, min(1, self.max_partition_size-1)))]}

        partitioned_terms = {}
        coeffs = [abs(coeff) for coeff, _ in cost_terms]
        max_coeff = max(coeffs) if coeffs else 1.0

        # Initialize partitions
        for i in range(self.n_partitions):
            partitioned_terms[i] = []

        # Process terms
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
            if i not in partitioned_terms:
                start_idx = i * self.max_partition_size
                end_idx = min(start_idx + self.max_partition_size, self.n_qubits)
                if end_idx - start_idx > 1:
                    partitioned_terms[i] = [(1.0, (0, 1))]

        logger.info(f"Partitioned cost terms into {len(partitioned_terms)} groups")
        return partitioned_terms

    def optimize(self, cost_terms: List[Tuple], steps: int = 100):
        """QAOA optimization with enhanced parameter validation."""
        try:
            # Initialize parameters - strictly 2 parameters for single layer
            params = np.array([
                np.random.uniform(-0.1, 0.1),      # gamma
                np.random.uniform(np.pi/2 - 0.1, np.pi/2 + 0.1)  # beta
            ])

            logger.info("Initial parameters: %s", params)

            opt = qml.AdamOptimizer(stepsize=0.05)
            costs = []
            best_cost = float('inf')
            best_params = None
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
                    logger.error("Error in cost function: %s", str(e))
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
                        logger.info("Step %d: Cost = %.6f", step, current_cost)
                    else:
                        no_improvement_count += 1

                    if no_improvement_count >= 10:
                        logger.info("Early stopping triggered at step %d", step)
                        break

                    # Update parameters with validation
                    params = self._validate_and_truncate_params(
                        opt.apply_grad(cost_function, params)
                    )

                except Exception as e:
                    logger.error("Error in optimization step %d: %s", step, str(e))
                    continue

            final_params = self._validate_and_truncate_params(
                best_params if best_params is not None else params
            )
            logger.info("Optimization complete with parameters: %s", final_params)

            return final_params, costs

        except Exception as e:
            logger.error("Error during optimization: %s", str(e))
            raise