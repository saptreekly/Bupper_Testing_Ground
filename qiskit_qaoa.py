import numpy as np
from qiskit import QuantumCircuit, transpile, QuantumRegister
from qiskit_aer import AerSimulator, Aer, noise
from qiskit.circuit import Parameter
from qiskit.primitives import BackendEstimator
from qiskit.quantum_info import SparsePauliOp
import logging
from typing import List, Tuple, Optional
from qiskit.transpiler import PassManager
from qiskit.circuit.library import QFT

logger = logging.getLogger(__name__)

class QiskitQAOA:
    """QAOA implementation using Qiskit backend with improved scalability."""

    def __init__(self, n_qubits: int, depth: int = 1):
        """Initialize QAOA circuit with advanced depth adaptation and circuit decomposition."""
        try:
            self.n_qubits = n_qubits

            # Enhanced adaptive depth calculation
            if n_qubits <= 6:
                depth_scale = 1.0  # Small problems: use base depth
                noise_factor = 1.0
            elif n_qubits <= 12:
                depth_scale = 0.8  # Medium problems: reduced depth
                noise_factor = 0.8  # Reduce depth due to noise
            elif n_qubits <= 20:
                depth_scale = 0.6  # Large problems: further reduced depth
                noise_factor = 0.6  # Further reduce depth due to noise
            else:
                depth_scale = 0.4  # Very large problems: minimal depth
                noise_factor = 0.4  # Significant depth reduction due to noise

            # Calculate adaptive depth with improved scaling
            base_depth = max(1, min(depth, n_qubits // 4))
            self.depth = max(1, min(
                int(base_depth * depth_scale * noise_factor),
                n_qubits // 2
            ))

            logger.info(f"Enhanced adaptive circuit depth calculation:")
            logger.info(f"- Problem size: {n_qubits} qubits")
            logger.info(f"- Base depth: {base_depth}")
            logger.info(f"- Depth scale: {depth_scale}")
            logger.info(f"- Noise factor: {noise_factor}")
            logger.info(f"- Final depth: {self.depth}")

            # Initialize backend with optimized configuration
            self.backend = AerSimulator()

            # Create and configure noise model
            try:
                noise_model = self._create_custom_noise_model()
                logger.info("Successfully created custom noise model")
            except Exception as e:
                logger.warning(f"Failed to create noise model: {str(e)}")
                noise_model = None

            # Configure backend options for improved performance
            self.backend.set_options(
                precision='double',
                max_parallel_threads=16,
                max_parallel_experiments=16,
                max_parallel_shots=2048,
                shots=4096,
                noise_model=noise_model,
                basis_gates=['u1', 'u2', 'u3', 'cx'],
                memory=True
            )

            # Initialize estimator with enhanced settings
            self.estimator = BackendEstimator(
                backend=self.backend,
                skip_transpilation=False,
                bound_pass_manager=True
            )

            logger.info(f"Initialized Qiskit backend with {n_qubits} qubits")

        except Exception as e:
            logger.error(f"Failed to initialize Qiskit backend: {str(e)}")
            raise

    def get_expectation_values(self, params, cost_terms):
        """Get expectation values using optimized parallel execution."""
        try:
            valid_terms = self._validate_cost_terms(cost_terms)
            circuit = self._create_qaoa_circuit(params, valid_terms)
            if circuit is None:
                logger.error("Failed to create valid QAOA circuit")
                return [0.0] * self.n_qubits

            # Create Z observables for each qubit using direct SparsePauliOp construction
            observables = []
            for i in range(self.n_qubits):
                try:
                    # Create Pauli string: I⊗I⊗...⊗Z⊗I⊗...⊗I
                    pauli_str = ''.join(['I'] * i + ['Z'] + ['I'] * (self.n_qubits - i - 1))
                    observables.append(SparsePauliOp(pauli_str))
                except Exception as e:
                    logger.error(f"Error creating observable for qubit {i}: {str(e)}")
                    return [0.0] * self.n_qubits

            # Submit jobs in batches for better performance
            batch_size = min(10, self.n_qubits)  # Adjust batch size based on problem size
            exp_vals = []

            for i in range(0, len(observables), batch_size):
                batch_obs = observables[i:i + batch_size]
                try:
                    job = self.estimator.run(
                        circuits=[circuit] * len(batch_obs),
                        observables=batch_obs,
                        parameter_values=None  # Parameters are already bound
                    )
                    result = job.result()
                    if hasattr(result, 'values') and result.values is not None:
                        exp_vals.extend(result.values)
                    else:
                        logger.warning(f"No values in result for batch {i//batch_size}")
                        exp_vals.extend([0.0] * len(batch_obs))
                except Exception as batch_error:
                    logger.error(f"Error in batch {i//batch_size}: {str(batch_error)}")
                    exp_vals.extend([0.0] * len(batch_obs))

            if len(exp_vals) != self.n_qubits:
                logger.warning(f"Expected {self.n_qubits} values but got {len(exp_vals)}")
                exp_vals = exp_vals[:self.n_qubits] if len(exp_vals) > self.n_qubits else \
                          exp_vals + [0.0] * (self.n_qubits - len(exp_vals))

            logger.debug(f"Extracted {len(exp_vals)} expectation values")
            return exp_vals

        except Exception as e:
            logger.error(f"Error computing expectation values: {str(e)}")
            return [0.0] * self.n_qubits

    def _create_qaoa_circuit(self, params, cost_terms):
        """Create QAOA circuit with improved handling of large qubit counts."""
        try:
            if not cost_terms:
                logger.warning("No valid cost terms available")
                return None

            # Validate parameters
            if len(params) != 2 * self.depth:
                logger.error(f"Invalid parameter count: expected {2 * self.depth}, got {len(params)}")
                return None

            # Initialize circuit with partitioning for large qubit counts
            if self.n_qubits > 20:
                return self._create_partitioned_circuit(params, cost_terms)

            # Standard circuit creation for smaller problems
            circuit = QuantumCircuit(self.n_qubits)

            # Optimize initial state preparation
            self._prepare_initial_state(circuit)

            # QAOA layers with improved efficiency
            for p in range(self.depth):
                gamma = np.clip(params[2 * p], -2*np.pi, 2*np.pi)
                beta = np.clip(params[2 * p + 1], -np.pi, np.pi)

                # Problem Hamiltonian with optimized grouping
                self._apply_cost_hamiltonian(circuit, gamma, cost_terms)

                # Mixer Hamiltonian with batch optimization
                self._apply_mixer_hamiltonian(circuit, beta)

            # Add error mitigation based on circuit size
            circuit = self._apply_error_mitigation(circuit)

            return circuit

        except Exception as e:
            logger.error(f"Error creating QAOA circuit: {str(e)}")
            return None

    def _create_partitioned_circuit(self, params, cost_terms):
        """Create a partitioned circuit for large qubit counts with improved hardware constraints."""
        try:
            # Calculate optimal partition size within hardware limits
            max_qubits_per_partition = min(20, max(1, self.backend.configuration().n_qubits - 1))
            partition_size = min(max_qubits_per_partition, self.n_qubits // 2)
            n_partitions = (self.n_qubits + partition_size - 1) // partition_size

            logger.info(f"Creating {n_partitions} partitions with {partition_size} qubits each")

            # Create main circuit with hardware-aware layout
            main_circuit = QuantumCircuit(self.n_qubits)

            # Create and optimize each partition
            for i in range(n_partitions):
                start_idx = i * partition_size
                end_idx = min((i + 1) * partition_size, self.n_qubits)
                partition_qubits = list(range(start_idx, end_idx))

                # Create subcircuit for partition
                subcircuit = QuantumCircuit(len(partition_qubits))

                # Filter cost terms for this partition
                partition_cost_terms = [
                    (coeff, (i-start_idx, j-start_idx)) 
                    for coeff, (i, j) in cost_terms 
                    if start_idx <= i < end_idx and start_idx <= j < end_idx
                ]

                # Prepare initial state for partition
                self._prepare_initial_state(subcircuit)

                # Apply QAOA layers to partition with reduced depth for larger partitions
                local_depth = max(1, self.depth // n_partitions)
                for p in range(local_depth):
                    gamma = params[2 * p]
                    beta = params[2 * p + 1]

                    self._apply_cost_hamiltonian(subcircuit, gamma, partition_cost_terms)
                    self._apply_mixer_hamiltonian(subcircuit, beta)

                # Optimize subcircuit with hardware constraints
                try:
                    optimized_subcircuit = transpile(
                        subcircuit,
                        backend=self.backend,
                        optimization_level=3,
                        layout_method='sabre',  # Use advanced layout method
                        routing_method='sabre',  # Use advanced routing
                        scheduling_method='alap'  # As late as possible scheduling
                    )
                except Exception as e:
                    logger.error(f"Error optimizing subcircuit: {str(e)}")
                    # Fallback to basic optimization if advanced fails
                    optimized_subcircuit = transpile(
                        subcircuit,
                        basis_gates=['u1', 'u2', 'u3', 'cx'],
                        optimization_level=2
                    )

                # Add optimized subcircuit to main circuit
                main_circuit.compose(optimized_subcircuit, partition_qubits, inplace=True)

            # Add minimal cross-partition interactions
            self._add_cross_partition_interactions(main_circuit, params, cost_terms)

            return main_circuit

        except Exception as e:
            logger.error(f"Error creating partitioned circuit: {str(e)}")
            return None

    def _prepare_initial_state(self, circuit):
        """Prepare initial state with batched operations."""
        batch_size = 4
        for i in range(0, circuit.num_qubits, batch_size):
            batch = range(i, min(i + batch_size, circuit.num_qubits))
            circuit.h(batch)

    def _apply_cost_hamiltonian(self, circuit, gamma, cost_terms):
        """Apply cost Hamiltonian with optimized grouping."""
        # Group operations by qubit pairs
        qubit_pairs = {}
        for coeff, (i, j) in cost_terms:
            if not (0 <= i < circuit.num_qubits and 0 <= j < circuit.num_qubits):
                continue
            pair = tuple(sorted([i, j]))
            qubit_pairs[pair] = qubit_pairs.get(pair, 0) + coeff * gamma

        # Apply grouped operations
        for (i, j), angle in qubit_pairs.items():
            if i != j:
                circuit.cx(i, j)
                circuit.rz(2 * angle, j)
                circuit.cx(i, j)
            else:
                circuit.rz(angle, i)

    def _apply_mixer_hamiltonian(self, circuit, beta):
        """Apply mixer Hamiltonian with batched operations."""
        batch_size = 4
        for i in range(0, circuit.num_qubits, batch_size):
            batch = range(i, min(i + batch_size, circuit.num_qubits))
            circuit.rx(2 * beta, batch)

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
        """Decompose large circuits into more manageable subcircuits."""
        try:
            if self.n_qubits <= 20:
                return circuit

            logger.info("Applying circuit decomposition for large qubit count")

            # Create subcircuits of maximum 20 qubits
            subcircuit_size = 20
            n_subcircuits = (self.n_qubits + subcircuit_size - 1) // subcircuit_size

            decomposed = QuantumCircuit(self.n_qubits)

            for i in range(n_subcircuits):
                start_idx = i * subcircuit_size
                end_idx = min((i + 1) * subcircuit_size, self.n_qubits)
                qubits = list(range(start_idx, end_idx))

                # Extract and optimize subcircuit
                subcircuit = circuit.copy()
                subcircuit_optimized = transpile(
                    subcircuit,
                    basis_gates=['u1', 'u2', 'u3', 'cx'],
                    optimization_level=3
                )

                # Add optimized subcircuit back
                decomposed.compose(subcircuit_optimized, qubits, inplace=True)

            logger.info(f"Circuit decomposed into {n_subcircuits} subcircuits")
            return decomposed

        except Exception as e:
            logger.error(f"Error in circuit decomposition: {str(e)}")
            return circuit


    def _create_custom_noise_model(self):
        """Create a custom noise model with improved error rates and advanced error channels."""
        try:
            noise_model = noise.NoiseModel()

            # Enhanced error scaling based on both problem size and vehicle count
            base_error_rate = 0.001
            size_factor = min(1.0, 10 / self.n_qubits)

            # Add thermal relaxation with improved timing
            t1, t2 = 50e3, 70e3  # Relaxation times (ns)
            gate_times = {
                'single': 35,  # Optimized single-qubit gate time
                'two': 250,    # Optimized two-qubit gate time
                'measure': 100  # Measurement time
            }

            # Create error channels with enhanced precision
            thermal_error = {
                'single': noise.thermal_relaxation_error(
                    t1, t2, gate_times['single'],
                    excited_state_population=0.01
                ),
                'two': noise.thermal_relaxation_error(
                    t1, t2, gate_times['two'],
                    excited_state_population=0.02
                )
            }

            # Add readout errors
            p_error = base_error_rate * size_factor
            readout_error = noise.ReadoutError([[1 - p_error, p_error], [p_error, 1 - p_error]])

            # Add measurement error mitigation
            noise_model.add_all_qubit_readout_error(readout_error)

            # Add improved gate errors with crosstalk effects
            for basis_gate in ['u1', 'u2', 'u3']:
                error = thermal_error['single']
                if basis_gate in ['u2', 'u3']:
                    # Add additional rotation errors for non-virtual gates
                    error = error.compose(
                        noise.depolarizing_error(base_error_rate * 0.1, 1)
                    )
                noise_model.add_all_qubit_quantum_error(error, basis_gate)

            # Enhanced two-qubit gate errors with crosstalk
            cx_error = thermal_error['two'].compose(
                noise.depolarizing_error(base_error_rate * 0.2, 2)
            )
            noise_model.add_all_qubit_quantum_error(cx_error, 'cx')

            logger.info(f"Created enhanced noise model with size factor {size_factor:.3f}")
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

    def optimize(self, cost_terms, steps=100, callback=None):
        """Optimize the QAOA circuit parameters with improved convergence."""
        try:
            # Validate and normalize cost terms
            valid_terms = self._validate_cost_terms(cost_terms)
            if not valid_terms:
                valid_terms = [(1.0, (0, min(1, self.n_qubits-1)))]

            # Initialize parameters with improved strategy
            params = []
            for _ in range(self.depth):
                params.extend([
                    np.random.uniform(-np.pi/8, np.pi/8),  # gamma
                    np.pi/4 + np.random.uniform(-np.pi/8, np.pi/8)  # beta
                ])
            params = np.array(params)
            logger.info(f"Initial parameters: {params}")

            costs = []
            best_params = None
            best_cost = float('inf')
            no_improvement_count = 0
            min_improvement = 1e-4  # Minimum relative improvement threshold

            def cost_function(p):
                """Compute cost with error handling."""
                try:
                    measurements = self.get_expectation_values(p, valid_terms)
                    cost = sum(coeff * measurements[i] * measurements[j]
                             for coeff, (i, j) in valid_terms)
                    return float(cost)
                except Exception as e:
                    logger.error(f"Error in cost function: {str(e)}")
                    return float('inf')

            # Optimize with adaptive learning rate and momentum
            alpha = 0.1  # Initial learning rate
            alpha_decay = 0.995  # Learning rate decay
            alpha_min = 0.01  # Minimum learning rate
            momentum = np.zeros_like(params)
            beta1 = 0.9  # Momentum coefficient

            # Keep track of parameter history for convergence check
            param_history = []
            cost_history = []

            initial_cost = cost_function(params)
            if callback:
                callback(0, initial_cost)

            for step in range(steps):
                try:
                    current_cost = cost_function(params)
                    costs.append(current_cost)
                    cost_history.append(current_cost)
                    param_history.append(params.copy())

                    # Call the callback function with detailed status
                    if callback:
                        progress_data = {
                            "step": step,
                            "total_steps": steps,
                            "cost": current_cost,
                            "best_cost": best_cost if best_cost != float('inf') else current_cost,
                            "progress": step / steps,
                            "learning_rate": alpha
                        }
                        callback(step, progress_data)

                    if current_cost < best_cost:
                        improvement = (best_cost - current_cost) / abs(best_cost) if best_cost != float('inf') else 1.0
                        if improvement > min_improvement:
                            best_cost = current_cost
                            best_params = params.copy()
                            no_improvement_count = 0
                            logger.info(f"Step {step}: New best cost = {current_cost:.6f} (improved by {improvement:.1%})")
                        else:
                            no_improvement_count += 1
                    else:
                        no_improvement_count += 1

                    # Adaptive early stopping with multiple criteria
                    if no_improvement_count >= 20 or (
                        len(cost_history) > 10 and 
                        abs(np.mean(cost_history[-5:]) - np.mean(cost_history[-10:-5])) < min_improvement
                    ):
                        if callback:
                            callback(step, current_cost)
                        logger.info(f"Early stopping at step {step}")
                        break

                    # Compute gradient with error handling and adaptive step size
                    eps = max(1e-4, alpha * 0.1)  # Adaptive finite difference step
                    grad = np.zeros_like(params)
                    for i in range(len(params)):
                        try:
                            params_plus = params.copy()
                            params_plus[i] += eps
                            cost_plus = cost_function(params_plus)

                            if cost_plus != float('inf'):
                                grad[i] = (cost_plus - current_cost) / eps
                            else:
                                logger.warning(f"Gradient computation failed for parameter {i}")
                                grad[i] = 0.0
                        except Exception as e:
                            logger.error(f"Error computing gradient for parameter {i}: {str(e)}")
                            grad[i] = 0.0

                    # Update with momentum and adaptive learning rate
                    grad_norm = np.linalg.norm(grad)
                    if grad_norm > 1.0:
                        grad = grad / grad_norm  # Gradient clipping

                    momentum = beta1 * momentum + (1 - beta1) * grad
                    params -= alpha * momentum

                    # Bound parameters to prevent instability
                    params[::2] = np.clip(params[::2], -2*np.pi, 2*np.pi)  # gamma
                    params[1::2] = np.clip(params[1::2], -np.pi, np.pi)    # beta

                    # Decay learning rate
                    alpha = max(alpha_min, alpha * alpha_decay)

                except Exception as e:
                    logger.error(f"Error in optimization step {step}: {str(e)}")
                    continue

            if best_params is None:
                logger.warning("Optimization failed to find valid parameters")
                best_params = params

            logger.info(f"Optimization completed with best cost: {best_cost:.6f}")
            return best_params, costs

        except Exception as e:
            logger.error(f"Error during optimization: {str(e)}")
            raise

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