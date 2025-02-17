import numpy as np
from typing import List, Tuple, Callable
import pennylane as qml
import logging

logger = logging.getLogger(__name__)

class QAOAOptimizer:
    def __init__(self, circuit_handler: Callable, n_params: int):
        """
        Initialize QAOA optimizer.

        Args:
            circuit_handler (Callable): Function that executes the quantum circuit
            n_params (int): Number of parameters to optimize
        """
        self.circuit_handler = circuit_handler
        self.n_params = n_params

    def optimize(self, cost_terms: List[Tuple], max_iterations: int = 100,
                learning_rate: float = 0.1, tolerance: float = 1e-5) -> Tuple[np.ndarray, List[float]]:
        """
        Optimize QAOA parameters using gradient descent.

        Args:
            cost_terms (List[Tuple]): Cost Hamiltonian terms
            max_iterations (int): Maximum number of optimization iterations
            learning_rate (float): Learning rate for gradient descent
            tolerance (float): Convergence tolerance

        Returns:
            Tuple[np.ndarray, List[float]]: Optimal parameters and cost history
        """
        # Initialize parameters with gradient support
        params = qml.numpy.array(np.random.uniform(0, 2*np.pi, self.n_params), requires_grad=True)
        optimizer = qml.GradientDescentOptimizer(stepsize=learning_rate)
        cost_history = []

        def cost_function(params):
            """Cost function that processes quantum measurements"""
            try:
                result = self.circuit_handler(params, cost_terms)
                logger.debug(f"Raw circuit output: {result}")

                # Handle measurement results
                if isinstance(result, list):
                    if len(result) == 1:
                        result = result[0]  # Extract single measurement
                    else:
                        result = np.mean(result)  # Average multiple measurements

                # Convert quantum types to numpy array if needed
                if hasattr(result, 'numpy'):
                    result = qml.numpy.array(result)

                logger.debug(f"Processed cost value: {result}")
                return result

            except Exception as e:
                logger.error(f"Error in cost function: {str(e)}")
                raise

        prev_cost = float('inf')
        try:
            for iteration in range(max_iterations):
                try:
                    # Optimization step
                    params = optimizer.step(cost_function, params)

                    # Calculate current cost
                    current_cost = cost_function(params)

                    # Convert to Python float for history
                    if hasattr(current_cost, 'numpy'):
                        history_cost = float(current_cost.numpy())
                    else:
                        history_cost = float(current_cost)

                    cost_history.append(history_cost)
                    logger.info(f"Iteration {iteration}: Cost = {history_cost:.6f}")

                    # Check convergence
                    if abs(prev_cost - history_cost) < tolerance:
                        logger.info("Optimization converged within tolerance")
                        break

                    prev_cost = history_cost

                except Exception as e:
                    logger.error(f"Error in optimization step {iteration}: {str(e)}")
                    raise

        except Exception as e:
            logger.error(f"Error during optimization: {str(e)}")
            raise

        return params, cost_history

    def compute_expectation(self, optimal_params: np.ndarray, 
                          cost_terms: List[Tuple]) -> float:
        """
        Compute expectation value with optimal parameters.

        Args:
            optimal_params (np.ndarray): Optimized parameters
            cost_terms (List[Tuple]): Cost Hamiltonian terms

        Returns:
            float: Expectation value
        """
        try:
            result = self.circuit_handler(optimal_params, cost_terms)
            if isinstance(result, list):
                result = result[0] if len(result) == 1 else np.mean(result)
            if hasattr(result, 'numpy'):
                return float(result.numpy())
            return float(result)
        except Exception as e:
            logger.error(f"Error computing expectation: {str(e)}")
            raise