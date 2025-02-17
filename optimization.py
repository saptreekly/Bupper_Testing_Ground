import numpy as np
from typing import List, Tuple, Callable
import pennylane as qml
import logging

logger = logging.getLogger(__name__)

class QAOAOptimizer:
    def __init__(self, circuit_handler: Callable, n_params: int):
        """Initialize the QAOA optimizer."""
        self.circuit_handler = circuit_handler
        self.n_params = n_params
        self.logger = logging.getLogger(__name__)

    def optimize(self, cost_terms: List[Tuple], max_iterations: int = 100,
                learning_rate: float = 0.05, tolerance: float = 1e-5) -> Tuple[np.ndarray, List[float]]:
        try:
            # Initialize parameters with gradient support
            params = qml.numpy.array(np.random.uniform(0, 2*np.pi, self.n_params), requires_grad=True)
            optimizer = qml.AdamOptimizer(stepsize=learning_rate)
            cost_history = []

            def cost_function(params):
                """Cost function that processes quantum measurements"""
                try:
                    # Get circuit measurement with batched parameters
                    measurements = self.circuit_handler(params, cost_terms)

                    # Calculate cost from measurements directly
                    cost = 0.0
                    for coeff, (i, j) in cost_terms:
                        if isinstance(measurements, (list, np.ndarray)):
                            cost += coeff * measurements[i] * measurements[j]
                        else:
                            # Handle single measurement case
                            cost += coeff * measurements

                    self.logger.debug("Calculated cost: %s", str(cost))
                    return cost

                except Exception as e:
                    self.logger.error("Error in cost function: %s", str(e))
                    raise

            prev_cost = float('inf')
            for iteration in range(max_iterations):
                try:
                    # One optimization step with parameter update
                    params, current_cost = optimizer.step_and_cost(cost_function, params)

                    # Convert cost for history (unwrap from ArrayBox if needed)
                    if hasattr(current_cost, 'numpy'):
                        history_cost = float(current_cost.numpy())
                    else:
                        history_cost = float(current_cost)

                    cost_history.append(history_cost)

                    # Log progress
                    self.logger.info("Iteration %d: Cost = %.6f", iteration, history_cost)

                    # Check convergence
                    if abs(prev_cost - history_cost) < tolerance:
                        self.logger.info("Optimization converged within tolerance")
                        break

                    prev_cost = history_cost

                except Exception as e:
                    self.logger.error("Error in optimization step %d: %s", iteration, str(e))
                    raise

            return params, cost_history

        except Exception as e:
            self.logger.error("Error during optimization: %s", str(e))
            raise

    def compute_expectation(self, optimal_params: np.ndarray, 
                          cost_terms: List[Tuple]) -> float:
        """
        Compute expectation value with optimal parameters.
        """
        try:
            measurements = self.circuit_handler(optimal_params, cost_terms)

            # Calculate final cost using measurements
            cost = 0.0
            for coeff, (i, j) in cost_terms:
                if isinstance(measurements, (list, np.ndarray)):
                    cost += coeff * measurements[i] * measurements[j]
                else:
                    cost += coeff * measurements

            if hasattr(cost, 'numpy'):
                cost = float(cost.numpy())
            return float(cost)

        except Exception as e:
            self.logger.error("Error computing expectation: %s", str(e))
            raise