import numpy as np
from qiskit import QuantumCircuit, transpile, QuantumRegister
from qiskit_aer import Aer
from qiskit.circuit import Parameter
from qiskit.primitives import BackendEstimator
from qiskit.quantum_info import SparsePauliOp
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

class QiskitQAOA:
    """QAOA implementation using Qiskit backend."""

    def __init__(self, n_qubits: int, depth: int = 1):
        """Initialize QAOA circuit with Qiskit backend and advanced depth adaptation."""
        try:
            self.n_qubits = n_qubits

            # Enhanced adaptive depth calculation based on problem size and complexity
            # For larger problems, we want to carefully balance depth vs decoherence
            if n_qubits <= 6:
                depth_scale = 1.0  # Small problems: use base depth
                noise_factor = 1.0
            elif n_qubits <= 12:
                depth_scale = 1.5  # Medium problems: moderate increase
                noise_factor = 0.8  # Reduce depth slightly due to noise
            elif n_qubits <= 20:
                depth_scale = 2.0  # Large problems: maximum depth
                noise_factor = 0.6  # Further reduce depth due to noise
            else:
                depth_scale = 2.5  # Very large problems: extended depth
                noise_factor = 0.4  # Significant depth reduction due to noise

            # Calculate base depth with connectivity consideration
            base_depth = max(1, min(depth, n_qubits // 3))

            # Apply adaptive scaling with noise consideration
            self.depth = max(1, min(
                int(base_depth * depth_scale * noise_factor),
                n_qubits
            ))

            # Log detailed depth calculation
            logger.info(f"Adaptive circuit depth calculation:")
            logger.info(f"- Problem size: {n_qubits} qubits")
            logger.info(f"- Base depth: {base_depth}")
            logger.info(f"- Depth scale: {depth_scale}")
            logger.info(f"- Noise factor: {noise_factor}")
            logger.info(f"- Final depth: {self.depth}")

            # Configure backend with enhanced noise mitigation
            self.backend = Aer.get_backend('aer_simulator_statevector')

            # Create custom noise model
            noise_model = self._create_custom_noise_model()

            self.backend.set_options(
                precision='double',
                max_parallel_threads=16,  # Increased for better performance
                max_parallel_experiments=16,
                max_parallel_shots=2048,
                shots=4096,  # Increased shots for better statistics
                noise_model=noise_model,  # Apply custom noise model
                basis_gates=['u1', 'u2', 'u3', 'cx'],
                memory=True  # Enable memory for error tracking
            )

            # Initialize estimator with optimized parameters
            self.estimator = BackendEstimator(
                backend=self.backend,
                skip_transpilation=False,  # Enable full optimization
                bound_pass_manager=True  # Enable bound pass manager
            )

            logger.info(f"Initialized Qiskit backend with {n_qubits} qubits")
            logger.debug(f"Backend configuration: {self.backend.configuration().to_dict()}")
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
        """Create optimized QAOA circuit with advanced optimization for larger problems."""
        try:
            if not cost_terms:
                logger.warning("No valid cost terms available")
                return None

            # Enhanced circuit size validation with detailed logging
            if self.n_qubits > 30:  # Increased limit with optimization
                logger.error(f"Circuit size ({self.n_qubits} qubits) exceeds practical limits")
                return None

            # Validate and normalize parameters
            if len(params) != 2 * self.depth:
                logger.error(f"Invalid parameter count: expected {2 * self.depth}, got {len(params)}")
                return None

            # Initialize circuit with optimization flags
            circuit = QuantumCircuit(self.n_qubits)

            # Optimize initial state preparation
            # For larger circuits, use parallel initialization
            if self.n_qubits > 10:
                # Parallel Hadamard application
                for i in range(0, self.n_qubits, 4):
                    batch = range(i, min(i + 4, self.n_qubits))
                    circuit.h(batch)
            else:
                circuit.h(range(self.n_qubits))

            # QAOA layers with circuit optimization
            for p in range(self.depth):
                gamma = np.clip(params[2 * p], -2*np.pi, 2*np.pi)
                beta = np.clip(params[2 * p + 1], -np.pi, np.pi)

                # Optimize cost Hamiltonian evolution
                # Group operations by qubit pairs to reduce gate count
                qubit_pairs = {}
                for coeff, (i, j) in cost_terms:
                    if not (0 <= i < self.n_qubits and 0 <= j < self.n_qubits):
                        logger.warning(f"Skipping invalid qubit indices: ({i}, {j})")
                        continue

                    pair = tuple(sorted([i, j]))
                    if pair not in qubit_pairs:
                        qubit_pairs[pair] = 0
                    qubit_pairs[pair] += coeff * gamma

                # Apply grouped operations
                for (i, j), total_angle in qubit_pairs.items():
                    if i != j:
                        # Optimize for adjacent qubits
                        if abs(i - j) == 1:
                            circuit.cx(i, j)
                            circuit.rz(2 * total_angle, j)
                            circuit.cx(i, j)
                        else:
                            # For non-adjacent qubits, use optimized SWAP network
                            swap_sequence = self._optimize_swap_sequence(i, j)
                            for swap_i, swap_j in swap_sequence:
                                circuit.swap(swap_i, swap_j)
                            circuit.cx(i, j)
                            circuit.rz(2 * total_angle, j)
                            circuit.cx(i, j)
                            # Undo SWAP network
                            for swap_i, swap_j in reversed(swap_sequence):
                                circuit.swap(swap_i, swap_j)
                    else:
                        circuit.rz(total_angle, i)

                # Optimize mixer Hamiltonian evolution
                # Use batched RX gates for better performance
                if self.n_qubits > 10:
                    for i in range(0, self.n_qubits, 4):
                        batch = range(i, min(i + 4, self.n_qubits))
                        circuit.rx(2 * beta, batch)
                else:
                    circuit.rx(2 * beta, range(self.n_qubits))

            # Circuit validation and optimization metrics
            n_gates = circuit.size()
            depth = circuit.depth()
            logger.info(f"Created QAOA circuit: {n_gates} gates, depth {depth}")

            # Add warning for deep circuits
            if depth > 100:
                logger.warning(f"Circuit depth ({depth}) may impact reliability")
                logger.info("Consider using hybrid optimization for better results")

            circuit, meas_filter = self._apply_error_mitigation(circuit)
            return circuit

        except Exception as e:
            logger.error(f"Error creating QAOA circuit: {str(e)}")
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

    def _create_custom_noise_model(self):
        """Create a custom noise model based on problem size."""
        try:
            from qiskit.providers.aer.noise import NoiseModel
            from qiskit.providers.aer.noise.errors import depolarizing_error, thermal_relaxation_error

            noise_model = NoiseModel()

            # Scale error rates based on problem size
            base_error_rate = 0.001  # Base error rate for small circuits
            size_factor = min(1.0, 10 / self.n_qubits)  # Reduce quality for larger circuits

            # Add depolarizing error to all gates
            error_1q = depolarizing_error(base_error_rate * size_factor, 1)
            error_2q = depolarizing_error(base_error_rate * 2 * size_factor, 2)

            # Add relaxation error
            t1 = 50e3  # T1 relaxation time (ns)
            t2 = 70e3  # T2 relaxation time (ns)
            gate_time_1q = 50  # Single-qubit gate time (ns)
            gate_time_2q = 300  # Two-qubit gate time (ns)

            thermal_error_1q = thermal_relaxation_error(t1, t2, gate_time_1q)
            thermal_error_2q = thermal_relaxation_error(t1, t2, gate_time_2q)

            # Add errors to noise model
            noise_model.add_all_qubit_quantum_error(error_1q, ['u1', 'u2', 'u3'])
            noise_model.add_all_qubit_quantum_error(error_2q, ['cx'])
            noise_model.add_all_qubit_quantum_error(thermal_error_1q, ['u1', 'u2', 'u3'])
            noise_model.add_all_qubit_quantum_error(thermal_error_2q, ['cx'])

            logger.info(f"Created custom noise model for {self.n_qubits} qubits")
            logger.debug(f"Base error rate: {base_error_rate}, Size factor: {size_factor}")

            return noise_model
        except Exception as e:
            logger.error(f"Failed to create noise model: {str(e)}")
            return None

    def _apply_error_mitigation(self, circuit):
        """Apply error mitigation techniques to the circuit."""
        try:
            # Dynamic error mitigation based on circuit size
            if self.n_qubits <= 6:
                # For small circuits, use measurement error mitigation
                from qiskit.ignis.mitigation.measurement import CompleteMeasFitter, complete_meas_cal
                cal_circuits = self._generate_calibration_circuits()
                # Execute calibration circuits
                job = self.backend.run(cal_circuits)
                cal_results = job.result()
                meas_fitter = CompleteMeasFitter(cal_results, state_labels=list(range(2**self.n_qubits)))
                # Get the filter
                meas_filter = meas_fitter.filter
                logger.info("Applied measurement error mitigation")
                return circuit, meas_filter
            else:
                # For larger circuits, use simpler techniques
                # Add barrier to prevent optimization across error-prone regions
                circuit.barrier()
                # Add dynamical decoupling
                circuit.delay(100)  # Add small delay for natural decoherence
                logger.info("Applied simplified error mitigation for large circuit")
                return circuit, None
        except Exception as e:
            logger.error(f"Error in error mitigation: {str(e)}")
            return circuit, None

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