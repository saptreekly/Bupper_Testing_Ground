import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)

class CircuitVisualizer:
    def __init__(self):
        """Initialize the circuit visualizer."""
        try:
            plt.style.use('default')  # Using default style instead of seaborn
            logger.info("Successfully initialized circuit visualizer")
        except Exception as e:
            logger.error(f"Failed to initialize visualizer: {str(e)}")
            raise

    def plot_circuit_results(self, measurements: List[float], title: str = "QAOA Results"):
        """
        Plot measurement results from the quantum circuit.

        Args:
            measurements (List[float]): Measurement results
            title (str): Plot title
        """
        try:
            plt.figure(figsize=(10, 6))
            plt.bar(range(len(measurements)), measurements)
            plt.xlabel('Qubit Index')
            plt.ylabel('Expectation Value')
            plt.title(title)
            plt.grid(True)
            plt.show()
            logger.info("Successfully plotted circuit results")
        except Exception as e:
            logger.error(f"Error plotting circuit results: {str(e)}")
            raise

    def plot_optimization_trajectory(self, costs: List[float]):
        """
        Plot optimization trajectory.

        Args:
            costs (List[float]): List of cost values during optimization
        """
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(costs, 'b-', label='Cost Value')
            plt.xlabel('Iteration')
            plt.ylabel('Cost')
            plt.title('QAOA Optimization Trajectory')
            plt.legend()
            plt.grid(True)
            plt.show()
            logger.info("Successfully plotted optimization trajectory")
        except Exception as e:
            logger.error(f"Error plotting optimization trajectory: {str(e)}")
            raise

    def plot_route(self, coordinates: List[Tuple[float, float]], route: List[int]):
        """
        Plot the optimized route.

        Args:
            coordinates (List[Tuple[float, float]]): City coordinates
            route (List[int]): Optimized route
        """
        try:
            plt.figure(figsize=(10, 10))

            # Plot cities
            x_coords = [coord[0] for coord in coordinates]
            y_coords = [coord[1] for coord in coordinates]
            plt.scatter(x_coords, y_coords, c='red', s=100)

            # Plot route
            for i in range(len(route)):
                start = coordinates[route[i]]
                end = coordinates[route[(i + 1) % len(route)]]
                plt.plot([start[0], end[0]], [start[1], end[1]], 'b-')

            # Add city labels
            for i, (x, y) in enumerate(coordinates):
                plt.annotate(f'City {i}', (x, y), xytext=(5, 5), textcoords='offset points')

            plt.title('Optimized Route')
            plt.xlabel('X Coordinate')
            plt.ylabel('Y Coordinate')
            plt.grid(True)
            plt.show()
            logger.info("Successfully plotted optimized route")
        except Exception as e:
            logger.error(f"Error plotting route: {str(e)}")
            raise