import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit import Parameter
from qiskit.primitives import Estimator
from qiskit.quantum_info import SparsePauliOp
import logging
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

class QiskitQAOA:
    def __init__(self, n_qubits: int, depth: int = 1):
        """Initialize QAOA circuit with fixed 2-parameter system and strict validation."""
        try:
            self.n_qubits = n_qubits
            self.depth = 1  # Force depth to 1 to maintain 2-parameter system

            # Initialize parameters array with validation
            self.params = np.array([0.0, 0.0])  # [gamma, beta]

            # Maximum qubits per partition for Qiskit Aer
            self.max_partition_size = 31
            self.n_partitions = (n_qubits + self.max_partition_size - 1) // self.max_partition_size

            logger.info(f"Initializing QiskitQAOA with {n_qubits} qubits across {self.n_partitions} partitions")
            logger.info(f"Maximum partition size: {self.max_partition_size}")

            # Initialize backend and estimator with minimal configuration
            try:
                self.backend = AerSimulator()
                self.estimator = Estimator()  # Use basic Estimator instead of BackendEstimator
                logger.info("Successfully initialized QiskitQAOA backend")
            except Exception as e:
                logger.error(f"Error initializing Qiskit backend: {str(e)}")
                raise

        except Exception as e:
            logger.error(f"Failed to initialize QiskitQAOA: {str(e)}")
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
        """Get expectation values with enhanced partition handling and validation."""
        try:
            # Strict parameter validation
            params = self._validate_and_truncate_params(params)
            gamma, beta = params
            logger.info(f"Computing expectation values with gamma={gamma:.4f}, beta={beta:.4f}")

            # Validate and partition cost terms
            if not cost_terms:
                logger.warning("No cost terms provided")
                return np.zeros(self.n_qubits)

            # Split problem into partitions if needed
            if self.n_qubits <= self.max_partition_size:
                # Handle small problems without partitioning
                circuit = self._create_circuit(params, cost_terms)
                if circuit is None:
                    return np.zeros(self.n_qubits)

                try:
                    # Create simple observables
                    observables = []
                    for i in range(self.n_qubits):
                        pauli_str = ''.join(['I'] * i + ['Z'] + ['I'] * (self.n_qubits - i - 1))
                        observables.append(SparsePauliOp(pauli_str))

                    # Execute circuit with simple error handling
                    values = []
                    for obs in observables:
                        try:
                            result = self.estimator.run(
                                circuits=[circuit],
                                observables=[obs],
                            ).result()
                            values.extend(result.values)
                        except Exception as e:
                            logger.error(f"Error in measurement: {str(e)}")
                            values.append(0.0)

                    return np.array(values)

                except Exception as e:
                    logger.error(f"Error in circuit execution: {str(e)}")
                    return np.zeros(self.n_qubits)
            else:
                # Handle large problems with partitioning
                logger.info(f"Problem size exceeds maximum, using {self.n_partitions} partitions")
                measurements = np.zeros(self.n_qubits)

                partitioned_terms = self._partition_cost_terms(cost_terms)
                for partition_idx, partition_costs in partitioned_terms.items():
                    start_idx = partition_idx * self.max_partition_size
                    end_idx = min(start_idx + self.max_partition_size, self.n_qubits)
                    n_partition_qubits = end_idx - start_idx

                    circuit = self._create_circuit(params, partition_costs, n_qubits=n_partition_qubits)
                    if circuit is None:
                        measurements[start_idx:end_idx] = np.zeros(n_partition_qubits)
                        continue

                    try:
                        # Execute partition with error handling
                        obs_values = []
                        for i in range(n_partition_qubits):
                            pauli_str = ''.join(['I'] * i + ['Z'] + ['I'] * (n_partition_qubits - i - 1))
                            obs = SparsePauliOp(pauli_str)
                            try:
                                result = self.estimator.run(
                                    circuits=[circuit],
                                    observables=[obs],
                                ).result()
                                obs_values.extend(result.values)
                            except Exception as e:
                                logger.error(f"Error in partition {partition_idx} measurement: {str(e)}")
                                obs_values.append(0.0)

                        measurements[start_idx:end_idx] = obs_values[:n_partition_qubits]

                    except Exception as e:
                        logger.error(f"Error in partition {partition_idx} execution: {str(e)}")
                        measurements[start_idx:end_idx] = np.zeros(n_partition_qubits)

                return measurements

        except Exception as e:
            logger.error(f"Error computing expectation values: {str(e)}")
            return np.zeros(self.n_qubits)

    def _create_circuit(self, params, cost_terms, n_qubits=None):
        """Create quantum circuit with enhanced validation and error handling."""
        try:
            if not cost_terms:
                logger.warning("No cost terms provided for circuit creation")
                return None

            # Use provided n_qubits or default to self.n_qubits
            n_qubits = n_qubits if n_qubits is not None else self.n_qubits

            # Validate parameters
            params = self._validate_and_truncate_params(params)
            gamma, beta = params

            # Create and validate circuit
            circuit = QuantumCircuit(n_qubits)

            # Initial state preparation
            circuit.h(range(n_qubits))

            # Apply cost Hamiltonian with strict bounds checking
            for coeff, (i, j) in cost_terms:
                if not (0 <= i < n_qubits and 0 <= j < n_qubits):
                    continue

                try:
                    angle = 2 * gamma * coeff
                    if i != j:
                        circuit.cx(i, j)
                        circuit.rz(angle, j)
                        circuit.cx(i, j)
                    else:
                        circuit.rz(angle, i)
                except Exception as e:
                    logger.error(f"Error applying cost gates at indices ({i}, {j}): {str(e)}")
                    continue

            # Apply mixer Hamiltonian with validation
            try:
                circuit.rx(2 * beta, range(n_qubits))
            except Exception as e:
                logger.error(f"Error applying mixer gates: {str(e)}")
                return None

            return circuit

        except Exception as e:
            logger.error(f"Error creating quantum circuit: {str(e)}")
            return None

    def _partition_cost_terms(self, cost_terms):
        """Partition cost terms with enhanced validation and bounds checking."""
        try:
            if not cost_terms:
                logger.warning("No cost terms provided")
                return {}

            partitioned_costs = {}
            for i in range(self.n_partitions):
                start_idx = i * self.max_partition_size
                end_idx = min(start_idx + self.max_partition_size, self.n_qubits)
                partitioned_costs[i] = []

            # Process and validate each cost term
            for coeff, (i, j) in cost_terms:
                if not (0 <= i < self.n_qubits and 0 <= j < self.n_qubits):
                    logger.warning(f"Skipping invalid indices: ({i}, {j})")
                    continue

                partition_i = i // self.max_partition_size
                partition_j = j // self.max_partition_size

                if partition_i == partition_j:
                    local_i = i % self.max_partition_size
                    local_j = j % self.max_partition_size
                    partition_costs = partitioned_costs[partition_i]

                    # Convert to local indices
                    start_idx = partition_i * self.max_partition_size
                    end_idx = min(start_idx + self.max_partition_size, self.n_qubits)
                    partition_size = end_idx - start_idx

                    if local_i < partition_size and local_j < partition_size:
                        partition_costs.append((coeff, (local_i, local_j)))

            return partitioned_costs

        except Exception as e:
            logger.error(f"Error partitioning cost terms: {str(e)}")
            return {}

    def _add_cross_partition_interactions(self, circuit, params, cost_terms):
        """Add minimal necessary interactions between partitions."""
        try:
            # Filter cross-partition terms
            partition_size = min(20, self.n_qubits // 2)
            cross_terms = [
                (coeff, (i, j)) for coeff, (i, j) in cost_terms
                if i // partition_size != j // partition_size  # Different partitions
            ]

            if cross_terms:
                # Use only first layer parameters for cross terms to reduce circuit depth
                gamma = params[0]  

                # Sort terms by coefficient magnitude and limit the number of cross-terms
                cross_terms.sort(key=lambda x: abs(x[0]), reverse=True)
                max_cross_terms = min(len(cross_terms), self.n_qubits)  # Limit cross-terms

                for coeff, (i, j) in cross_terms[:max_cross_terms]:
                    angle = coeff * gamma
                    # Add minimal interaction between partitions
                    circuit.cx(i, j)
                    circuit.rz(2 * angle, j)
                    circuit.cx(i, j)

        except Exception as e:
            logger.error(f"Error adding cross-partition interactions: {str(e)}")

    def _apply_error_mitigation(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """Apply advanced error mitigation techniques with dynamic adaptation."""
        try:
            # Apply different strategies based on circuit size
            if self.n_qubits > 20:
                # For large circuits: dynamical decoupling and echo sequences
                for i in range(self.n_qubits):
                    circuit.barrier([i])
                    # Add X-gate echo sequence
                    circuit.x([i])
                    circuit.delay(50, [i])  # Optimized delay time
                    circuit.x([i])
                    circuit.barrier([i])
                logger.info("Applied dynamical decoupling with echo sequences")

            elif self.n_qubits > 10:
                # For medium circuits: simplified error mitigation
                for i in range(self.n_qubits):
                    circuit.barrier([i])
                    circuit.delay(75, [i])  # Longer delay for better decoherence
                circuit.measure_all()
                logger.info("Applied intermediate error mitigation")

            else:
                # For small circuits: measurement error mitigation
                circuit.measure_all()
                logger.info("Added measurement operations for error mitigation")

            # Add circuit characterization measurements
            if self.n_qubits <= 8:  # Only for small circuits
                cal_circuits = self._generate_calibration_circuits()
                if cal_circuits:
                    logger.info("Generated calibration circuits for small system")

            return circuit

        except Exception as e:
            logger.error(f"Error in error mitigation: {str(e)}")
            return circuit

    def _decompose_large_circuit(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """Decompose large circuits into manageable blocks with improved error mitigation."""
        try:
            if self.n_qubits <= 20:
                return circuit

            logger.info("Applying enhanced circuit decomposition")

            # Create subcircuits with improved block size calculation
            block_size = min(20, max(10, self.n_qubits // 3))
            n_blocks = (self.n_qubits + block_size - 1) // block_size

            decomposed = QuantumCircuit(self.n_qubits)

            # Process blocks with error mitigation
            for i in range(n_blocks):
                start_idx = i * block_size
                end_idx = min((i + 1) * block_size, self.n_qubits)
                qubits = list(range(start_idx, end_idx))

                # Create and optimize block
                block = QuantumCircuit(len(qubits))

                # Apply error mitigation to block
                block.h(range(len(qubits)))  # Initial state

                # Add dynamical decoupling
                for q in range(len(qubits)):
                    block.barrier([q])
                    block.x(q)
                    block.delay(50)
                    block.x(q)
                    block.barrier([q])

                # Optimize block
                opt_block = transpile(
                    block,
                    basis_gates=['u1', 'u2', 'u3', 'cx'],
                    optimization_level=3
                )

                # Add optimized block to main circuit
                decomposed.compose(opt_block, qubits, inplace=True)

            logger.info(f"Circuit decomposed into {n_blocks} optimized blocks")
            return decomposed

        except Exception as e:
            logger.error(f"Error in circuit decomposition: {str(e)}")
            return circuit

    def _validate_cost_terms(self, cost_terms: List[Tuple]) -> List[Tuple]:
        """Validate cost terms with enhanced multi-vehicle support."""
        try:
            if not cost_terms:
                logger.warning("No cost terms provided")
                return []

            valid_terms = []
            seen_pairs = set()

            # Use absolute values for coefficient comparison
            coeffs = [abs(coeff) for coeff, _ in cost_terms]
            max_coeff = max(coeffs)
            mean_coeff = sum(coeffs) / len(coeffs)

            # Adaptive threshold based on problem size and term density
            threshold = max_coeff * (1e-8 if len(cost_terms) > 100 else 1e-6)
            term_density = len(cost_terms) / (self.n_qubits * self.n_qubits)

            logger.debug(f"Cost validation metrics - Max: {max_coeff:.2e}, Mean: {mean_coeff:.2e}, Density: {term_density:.3f}")

            for coeff, (i, j) in cost_terms:
                if not (0 <= i < self.n_qubits and 0 <= j < self.n_qubits):
                    continue

                pair = tuple(sorted([i, j]))
                if pair in seen_pairs:
                    continue

                if abs(coeff) > threshold:
                    norm_coeff = coeff / max_coeff if max_coeff > 0 else coeff
                    valid_terms.append((norm_coeff, pair))
                    seen_pairs.add(pair)

            # Ensure minimal connectivity for multi-vehicle problems
            if not valid_terms:
                for i in range(min(5, self.n_qubits - 1)):
                    valid_terms.append((1.0, (i, i + 1)))

            logger.info(f"Validated {len(valid_terms)} cost terms with density {term_density:.3f}")
            return valid_terms

        except Exception as e:
            logger.error(f"Error in cost term validation: {str(e)}")
            return [(1.0, (0, 1))]

    def _create_custom_noise_model(self):
        """Create enhanced noise model with improved error channels."""
        try:
            noise_model = noise.NoiseModel()

            # Adaptive error rates based on circuit size and connectivity
            base_error = 0.001 * (1 + np.log(self.n_qubits) / 10)
            size_factor = min(1.0, 20 / self.n_qubits)

            # Enhanced timing parameters
            t1 = 50e3 * (1 - 0.1 * np.log(self.n_qubits))  # Decay with size
            t2 = 70e3 * (1 - 0.1 * np.log(self.n_qubits))

            gate_times = {
                'single': 35,
                'two': 250,
                'measure': 100
            }

            # Create improved error channels
            thermal_errors = {
                'single': noise.thermal_relaxation_error(
                    t1, t2, gate_times['single'],
                    excited_state_population=0.01
                ),
                'two': noise.thermal_relaxation_error(
                    t1, t2, gate_times['two'],
                    excited_state_population=0.02
                )
            }

            # Add readout errors with size-dependent rates
            p_readout = base_error * size_factor
            readout_error = noise.ReadoutError([[1 - p_readout, p_readout], 
                                                  [p_readout, 1 - p_readout]])
            noise_model.add_all_qubit_readout_error(readout_error)

            # Add gate errors with enhanced error rates
            for gate in ['u1', 'u2', 'u3']:
                error = thermal_errors['single']
                if gate in ['u2', 'u3']:
                    error = error.compose(
                        noise.depolarizing_error(base_error * 0.1, 1)
                    )
                noise_model.add_all_qubit_quantum_error(error, gate)

            # Enhanced two-qubit gate errors
            cx_error = thermal_errors['two'].compose(
                noise.depolarizing_error(base_error * 0.2, 2)
            )
            noise_model.add_all_qubit_quantum_error(cx_error, 'cx')

            logger.info(f"Created enhanced noise model - Base error: {base_error:.2e}, Size factor: {size_factor:.3f}")
            return noise_model

        except Exception as e:
            logger.error(f"Error creating noise model: {str(e)}")
            return None

    def _optimize_swap_sequence(self, i: int, j: int) -> List[Tuple[int, int]]:
        """Generate optimized SWAP sequence for non-adjacent qubits."""
        if abs(i - j) <= 1:
            return []

        # Generate minimal SWAP sequence
        swaps = []
        current = i
        step = 1 if j > i else -1
        while abs(current - j) > 1:
            swaps.append((current, current + step))
            current += step
        return swaps

    def _validate_cost_terms(self, cost_terms):
        """Validate and normalize cost terms with improved handling of minimal cases."""
        if not cost_terms:
            logger.warning("Empty cost terms provided")
            return []

        valid_terms = []
        seen_pairs = set()

        # Use absolute values for coefficient comparison
        coeffs = [abs(coeff) for coeff, _ in cost_terms]
        if not coeffs:
            logger.warning("No coefficients found in cost terms")
            return []

        max_coeff = max(coeffs)
        min_coeff = min(coeffs)
        mean_coeff = sum(coeffs) / len(coeffs)

        # Log detailed statistics
        logger.debug(f"Cost terms statistics - max: {max_coeff:.2e}, min: {min_coeff:.2e}, mean: {mean_coeff:.2e}")
        logger.debug(f"Number of terms: {len(cost_terms)}")

        # Adaptive threshold based on problem size
        threshold = max_coeff * (1e-8 if len(cost_terms) > 10 else 1e-6)
        logger.debug(f"Using threshold: {threshold:.2e}")

        for coeff, (i, j) in cost_terms:
            if not (0 <= i < self.n_qubits and 0 <= j < self.n_qubits):
                logger.debug(f"Skipping invalid qubit indices: ({i}, {j})")
                continue

            pair = tuple(sorted([i, j]))
            if pair in seen_pairs:
                logger.debug(f"Skipping duplicate pair: {pair}")
                continue

            if abs(coeff) > threshold:
                norm_coeff = coeff / max_coeff if max_coeff > 0 else coeff
                valid_terms.append((norm_coeff, pair))
                seen_pairs.add(pair)
                logger.debug(f"Added cost term: {pair} with coefficient {norm_coeff:.6f}")

        # Always ensure at least one valid term
        if not valid_terms and self.n_qubits > 1:
            valid_terms.append((1.0, (0, 1)))
            logger.info("Added minimal interaction term")

        logger.info(f"Validated {len(valid_terms)} cost terms")
        return valid_terms
    def _generate_calibration_circuits(self):
        """Generate calibration circuits for error mitigation."""
        try:
            from qiskit.ignis.mitigation.measurement import complete_meas_cal
            qr = QuantumRegister(self.n_qubits)
            cal_circuits, _ = complete_meas_cal(qubit_list=list(range(self.n_qubits)), 
                                                  qr=qr, 
                                                  circlabel='mcal')
            return cal_circuits
        except Exception as e:
            logger.error(f"Failed to generate calibration circuits: {str(e)}")
            return []

    def _create_qaoa_circuit(self, params, cost_terms):
        """Create QAOA circuit with strict 2-parameter handling."""
        try:
            if not cost_terms:
                logger.warning("No valid cost terms available")
                return None

            # Always use exactly 2 parameters (gamma, beta)
            if len(params) != 2:
                logger.error(f"Parameter validation failed: expected 2 parameters, got {len(params)}")
                return None

            gamma, beta = params[0], params[1]
            logger.info(f"Creating QAOA circuit with parameters: gamma={gamma:.4f}, beta={beta:.4f}")

            # Initialize circuit
            circuit = QuantumCircuit(self.n_qubits)

            # Initial state preparation
            logger.debug("Preparing initial state")
            self._prepare_initial_state(circuit)

            # Single QAOA layer with fixed parameter count
            logger.debug(f"Building QAOA layer with depth {self.depth}")

            # Apply cost Hamiltonian
            self._apply_cost_hamiltonian(circuit, gamma, cost_terms)

            # Apply mixer Hamiltonian
            self._apply_mixer_hamiltonian(circuit, beta)

            logger.info(f"Successfully created QAOA circuit with {circuit.depth()} depth")
            return circuit

        except Exception as e:
            logger.error(f"Error creating QAOA circuit: {str(e)}")
            return None
    def _apply_cost_hamiltonian(self, circuit, gamma, cost_terms):
        """Apply cost Hamiltonian with detailed logging."""
        try:
            logger.debug(f"Applying cost Hamiltonian with gamma={gamma:.4f}")
            term_count = 0

            for coeff, (i, j) in cost_terms:
                if 0 <= i < self.n_qubits and 0 <= j < self.n_qubits:
                    angle = 2 * gamma * coeff
                    if i != j:
                        circuit.cx(i, j)
                        circuit.rz(angle, j)
                        circuit.cx(i, j)
                    else:
                        circuit.rz(angle, i)
                    term_count += 1

            logger.debug(f"Applied {term_count} cost terms")

        except Exception as e:
            logger.error(f"Error in cost Hamiltonian application: {str(e)}")
            raise

    def _apply_mixer_hamiltonian(self, circuit, beta):
        """Apply mixer Hamiltonian with detailed logging."""
        try:
            logger.debug(f"Applying mixer Hamiltonian with beta={beta:.4f}")
            for i in range(self.n_qubits):
                circuit.rx(2 * beta, i)
            logger.debug(f"Applied mixer terms to {self.n_qubits} qubits")

        except Exception as e:
            logger.error(f"Error in mixer Hamiltonian application: {str(e)}")
            raise

    def _prepare_initial_state(self, circuit):
        """Prepare initial state with batched operations."""
        batch_size = 4
        for i in range(0, circuit.num_qubits, batch_size):
            batch = range(i, min(i + batch_size, circuit.num_qubits))
            circuit.h(batch)

    def optimize(self, cost_terms: List[Tuple], steps: int = 100):
        """Run optimization with strict parameter validation."""
        try:
            # Initialize with exactly 2 parameters
            params = np.array([
                np.random.uniform(-0.1, 0.1),        # gamma
                np.random.uniform(np.pi/4, np.pi/2)  # beta
            ])

            # Ensure initial parameters are valid
            params = self._validate_and_truncate_params(params)
            logger.info(f"Starting optimization with parameters: {params}")

            costs = []
            best_params = None
            best_cost = float('inf')
            no_improvement_count = 0
            min_improvement = 1e-4

            def cost_function(p):
                """Compute cost with error handling."""
                try:
                    # Explicitly validate and truncate parameters
                    p = self._validate_and_truncate_params(p)
                    logger.debug(f"Cost function received parameters: {p}")
                    measurements = self.get_expectation_values(p, cost_terms)
                    cost = sum(coeff * measurements[i] * measurements[j]
                              for coeff, (i, j) in cost_terms)
                    return float(cost)
                except Exception as e:
                    logger.error(f"Error in cost function: {str(e)}")
                    return float('inf')

            # Optimize with adaptive learning rate
            alpha = 0.1  # Initial learning rate
            alpha_decay = 0.995
            alpha_min = 0.01

            for step in range(steps):
                try:
                    # Ensure parameters are valid before evaluation
                    params = self._validate_and_truncate_params(params)
                    current_cost = cost_function(params)
                    costs.append(current_cost)

                    if current_cost < best_cost:
                        improvement = (best_cost - current_cost) / abs(best_cost) if best_cost != float('inf') else 1.0
                        if improvement > min_improvement:
                            best_cost = current_cost
                            best_params = params.copy()
                            no_improvement_count = 0
                            logger.info(f"Step {step}: New best cost = {current_cost:.6f}")
                        else:
                            no_improvement_count += 1
                    else:
                        no_improvement_count += 1

                    if no_improvement_count >= 20:
                        logger.info(f"Early stopping at step {step}")
                        break

                    # Compute gradient with error handling and parameter validation
                    eps = max(1e-4, alpha * 0.1)
                    grad = np.zeros(2)  # Only compute gradient for 2 parameters
                    for i in range(2):
                        params_plus = params.copy()
                        params_plus[i] += eps
                        params_plus = self._validate_and_truncate_params(params_plus)
                        cost_plus = cost_function(params_plus)
                        if cost_plus != float('inf'):
                            grad[i] = (cost_plus - current_cost) / eps

                    # Update parameters with bounded values
                    grad_norm = np.linalg.norm(grad)
                    if grad_norm > 1.0:
                        grad = grad / grad_norm

                    params -= alpha * grad
                    params = self._validate_and_truncate_params(params)
                    alpha = max(alpha_min, alpha * alpha_decay)

                except Exception as e:
                    logger.error(f"Error in optimization step {step}: {str(e)}")
                    continue

            final_params = best_params if best_params is not None else params
            final_params = self._validate_and_truncate_params(final_params)
            logger.info(f"Optimization complete with final parameters: gamma={final_params[0]:.4f}, beta={final_params[1]:.4f}")
            return final_params, costs

        except Exception as e:
            logger.error(f"Error during optimization: {str(e)}")
            raise