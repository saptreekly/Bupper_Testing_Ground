import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from typing import List, Tuple, Set
import logging

logger = logging.getLogger(__name__)

class CircuitVisualizer:
    def __init__(self):
        """Initialize the circuit visualizer."""
        try:
            plt.style.use('default')
            # Set non-interactive backend
            plt.switch_backend('Agg')
            logger.info("Successfully initialized circuit visualizer")
        except Exception as e:
            logger.error(f"Failed to initialize visualizer: {str(e)}")
            raise

    def plot_optimization_trajectory(self, costs: List[float], save_path: str = "optimization_trajectory.png"):
        """
        Plot and save optimization trajectory.
        """
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(costs, 'b-', label='Cost Value')
            plt.xlabel('Iteration')
            plt.ylabel('Cost')
            plt.title('QAOA Optimization Trajectory')
            plt.legend()
            plt.grid(True)
            plt.savefig(save_path)
            plt.close()
            logger.info(f"Optimization trajectory plot saved to {save_path}")
        except Exception as e:
            logger.error(f"Error plotting optimization trajectory: {str(e)}")
            raise

    def plot_route(self, coordinates: List[Tuple[int, int]], route: List[int], 
                  grid_size: int = 16, obstacles: Set = None, save_path: str = "route.png"):
        """
        Plot and save the optimized route on a grid with obstacles.
        """
        try:
            plt.figure(figsize=(12, 12))

            # Draw grid
            for i in range(grid_size + 1):
                plt.axhline(y=i, color='gray', linestyle='-', alpha=0.3)
                plt.axvline(x=i, color='gray', linestyle='-', alpha=0.3)

            # Draw obstacles
            if obstacles:
                for obs_type, x, y in obstacles:
                    if obs_type == 'h':  # Horizontal obstacle
                        plt.plot([x, x+1], [y, y], 'r-', linewidth=3, alpha=0.7)
                    else:  # Vertical obstacle
                        plt.plot([x, x], [y, y+1], 'r-', linewidth=3, alpha=0.7)

            # Plot cities
            x_coords = [coord[0] for coord in coordinates]
            y_coords = [coord[1] for coord in coordinates]
            plt.scatter(x_coords, y_coords, c='red', s=200, zorder=5)

            # Plot route along grid lines
            for i in range(len(route)):
                start = coordinates[route[i]]
                end = coordinates[route[(i + 1) % len(route)]]

                # Draw path along grid lines (Manhattan path)
                plt.plot([start[0], start[0]], 
                        [start[1], end[1]], 
                        'b-', linewidth=2, zorder=2)
                plt.plot([start[0], end[0]], 
                        [end[1], end[1]], 
                        'b-', linewidth=2, zorder=2)

            # Add city labels
            for i, (x, y) in enumerate(coordinates):
                plt.annotate(f'City {i}', (x, y), 
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=12, fontweight='bold')

            plt.title('Optimized Route on Grid with Obstacles')
            plt.xlabel('X Coordinate')
            plt.ylabel('Y Coordinate')
            plt.grid(True)
            plt.axis('equal')
            plt.xlim(-1, grid_size+1)
            plt.ylim(-1, grid_size+1)
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
            logger.info(f"Route plot saved to {save_path}")
        except Exception as e:
            logger.error(f"Error plotting route: {str(e)}")
            raise

    def plot_circuit_results(self, measurements: List[float], title: str = "QAOA Results", save_path: str = "circuit_results.png"):
        """
        Plot and save measurement results from the quantum circuit.
        """
        try:
            plt.figure(figsize=(10, 6))
            plt.bar(range(len(measurements)), measurements)
            plt.xlabel('Qubit Index')
            plt.ylabel('Expectation Value')
            plt.title(title)
            plt.grid(True)
            plt.savefig(save_path)
            plt.close()
            logger.info(f"Circuit results plot saved to {save_path}")
        except Exception as e:
            logger.error(f"Error plotting circuit results: {str(e)}")
            raise