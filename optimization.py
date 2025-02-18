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
            # Initialize parameters
            params = np.array([0.01, 0.1], dtype=float)  # Start with small values
            optimizer = qml.GradientDescentOptimizer(stepsize=learning_rate)
            cost_history = []

            def cost_function(params):
                """Cost function that processes quantum measurements"""
                try:
                    # Get circuit measurements as numpy array
                    measurements = np.array(self.circuit_handler(params, cost_terms))

                    # Calculate cost from measurements
                    cost = 0.0
                    for coeff, (i, j) in cost_terms:
                        cost += float(coeff) * float(measurements[i]) * float(measurements[j])

                    return float(cost)

                except Exception as e:
                    self.logger.error("Error in cost function: %s", str(e))
                    raise

            prev_cost = float('inf')
            for iteration in range(max_iterations):
                try:
                    # One optimization step
                    params, current_cost = optimizer.step_and_cost(cost_function, params)
                    cost_history.append(float(current_cost))

                    # Log progress
                    self.logger.info("Iteration %d: Cost = %.6f", iteration, current_cost)

                    # Check convergence
                    if abs(prev_cost - current_cost) < tolerance:
                        self.logger.info("Optimization converged within tolerance")
                        break

                    prev_cost = current_cost

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
            # Get measurements as numpy array
            measurements = np.array(self.circuit_handler(optimal_params, cost_terms))

            # Calculate final cost using measurements
            cost = 0.0
            for coeff, (i, j) in cost_terms:
                cost += float(coeff) * float(measurements[i]) * float(measurements[j])

            return float(cost)

        except Exception as e:
            self.logger.error("Error computing expectation: %s", str(e))
            raise