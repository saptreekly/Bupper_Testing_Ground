import numpy as np
import logging
from typing import List, Tuple, Optional
from scipy.optimize import minimize
from qiskit_qaoa import QiskitQAOA
from qaoa_core import QAOACircuit

logger = logging.getLogger(__name__)

class HybridOptimizer:
    """Hybrid quantum-classical optimizer for QAOA with improved performance."""

    def __init__(self, n_qubits: int, depth: int = 1, backend: str = 'qiskit'):
        """Initialize hybrid optimizer with specified quantum backend."""
        self.n_qubits = n_qubits
        self.depth = depth
        self.backend = backend

        # Initialize quantum circuit based on backend choice
        if backend == 'qiskit':
            self.quantum_circuit = QiskitQAOA(n_qubits, depth)
        else:
            self.quantum_circuit = QAOACircuit(n_qubits, depth)

        logger.info(f"Initialized hybrid optimizer with {backend} backend")
        logger.debug(f"Configuration: {n_qubits} qubits, depth {depth}")

    def _classical_pre_optimization(self, cost_terms: List[Tuple]) -> np.ndarray:
        """Use classical optimization to find good initial parameters."""
        try:
            logger.info("Starting classical pre-optimization phase")

            def classical_cost(params):
                """Classical approximation of the quantum cost function."""
                cost = 0.0
                for coeff, (i, j) in cost_terms:
                    # Improved classical approximation using periodic functions
                    zi = np.cos(params[0]) * np.sin(params[1])
                    zj = np.cos(params[0]) * np.sin(params[1])
                    cost += coeff * zi * zj
                return float(cost)

            # Multiple starts with improved parameter ranges
            best_cost = float('inf')
            best_params = None
            n_starts = 5

            for start in range(n_starts):
                # Initialize with smaller parameter range for better convergence
                initial_guess = np.random.uniform(-np.pi/8, np.pi/8, 2 * self.depth)
                logger.debug(f"Start {start + 1}/{n_starts}: Initial parameters: {initial_guess}")

                result = minimize(classical_cost, initial_guess, 
                               method='COBYLA',
                               options={'maxiter': 25, 'rhobeg': 0.1})

                if result.fun < best_cost:
                    best_cost = result.fun
                    best_params = result.x
                    logger.debug(f"New best classical solution: cost = {best_cost:.6f}")

            if best_params is None:
                raise ValueError("Classical pre-optimization failed to find valid parameters")

            logger.info(f"Classical pre-optimization complete: cost = {best_cost:.6f}")
            return best_params

        except Exception as e:
            logger.error("Error in classical pre-optimization: %s", str(e))
            raise

    def optimize(self, cost_terms: List[Tuple], steps: int = 100) -> Tuple[np.ndarray, List[float]]:
        """Run hybrid optimization process with adaptive phase lengths."""
        try:
            # Phase 1: Classical pre-optimization
            initial_params = self._classical_pre_optimization(cost_terms)
            logger.info("Using classically optimized initial parameters")
            logger.debug(f"Initial parameters: {initial_params}")

            # Phase 2: Quantum optimization with adaptive steps
            remaining_steps = steps
            quantum_steps = min(remaining_steps, 50)  # Start with shorter quantum phase
            logger.info(f"Starting quantum optimization phase: {quantum_steps} steps")

            final_params, costs = self.quantum_circuit.optimize(cost_terms, steps=quantum_steps)
            current_cost = costs[-1]
            logger.info(f"Quantum optimization complete: cost = {current_cost:.6f}")

            # Phase 3: Classical refinement
            def quantum_cost(params):
                """Quantum cost function for classical refinement."""
                measurements = self.quantum_circuit.get_expectation_values(params, cost_terms)
                return sum(coeff * measurements[i] * measurements[j] 
                         for coeff, (i, j) in cost_terms)

            logger.info("Starting classical refinement phase")
            result = minimize(quantum_cost, final_params, 
                           method='COBYLA',
                           options={'maxiter': remaining_steps - quantum_steps,
                                  'rhobeg': 0.05})

            final_params = result.x
            costs.extend([result.fun] * (remaining_steps - quantum_steps))

            improvement = (current_cost - result.fun) / abs(current_cost)
            logger.info(f"Classical refinement improved cost by {improvement:.1%}")
            logger.info(f"Hybrid optimization complete: final cost = {result.fun:.6f}")

            return final_params, costs

        except Exception as e:
            logger.error("Error in hybrid optimization: %s", str(e))
            raise

    def get_expectation_values(self, params: np.ndarray, cost_terms: List[Tuple]) -> List[float]:
        """Get expectation values using the quantum circuit."""
        try:
            return self.quantum_circuit.get_expectation_values(params, cost_terms)
        except Exception as e:
            logger.error("Error getting expectation values: %s", str(e))
            raise