import numpy as np
from typing import List, Tuple, Callable
import pennylane as qml

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

    def initialize_parameters(self) -> np.ndarray:
        """
        Initialize QAOA parameters.

        Returns:
            np.ndarray: Initial parameters
        """
        # Initialize parameters with requires_grad=True
        params = np.random.uniform(0, 2*np.pi, self.n_params)
        return qml.numpy.array(params, requires_grad=True)

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
        params = self.initialize_parameters()
        optimizer = qml.GradientDescentOptimizer(stepsize=learning_rate)
        cost_history = []

        def objective(params):
            params = qml.numpy.array(params, requires_grad=True)
            return self.circuit_handler(params, cost_terms)

        prev_cost = float('inf')
        for iteration in range(max_iterations):
            # Optimization step
            params = optimizer.step(objective, params)

            # Calculate current cost
            current_cost = np.sum(objective(params))
            cost_history.append(float(current_cost))

            # Check convergence
            if abs(prev_cost - current_cost) < tolerance:
                break

            prev_cost = current_cost
            print(f"Iteration {iteration}: Cost = {current_cost:.4f}")

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
        optimal_params = qml.numpy.array(optimal_params, requires_grad=False)
        return float(np.sum(self.circuit_handler(optimal_params, cost_terms)))