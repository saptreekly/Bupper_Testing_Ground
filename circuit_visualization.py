import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from typing import List, Tuple, Set
import logging
from matplotlib.patches import Rectangle

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
        """Plot and save optimization trajectory."""
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
                   grid_size: int = 16, obstacles: Set = None, route_paths: List[List] = None,
                   save_path: str = "route.png"):
        """
        Plot and save the optimized route on a grid with obstacles.
        Shows grid-aligned paths (vertical then horizontal) between cities.
        """
        try:
            plt.figure(figsize=(12, 12))
            ax = plt.gca()

            # Draw grid
            for i in range(grid_size + 1):
                plt.axhline(y=i, color='gray', linestyle='-', alpha=0.2)
                plt.axvline(x=i, color='gray', linestyle='-', alpha=0.2)

            # Create set of blocked squares from obstacles
            blocked_squares = set()
            if obstacles:
                for obs_type, x, y in obstacles:
                    if obs_type == 'h':
                        blocked_squares.add((x, y-1))
                    else:  # vertical
                        blocked_squares.add((x-1, y))

            # Draw obstacles as solid blocks
            for x, y in blocked_squares:
                rect = Rectangle((x, y), 1, 1, facecolor='red', alpha=0.3)
                ax.add_patch(rect)

            # Plot cities with larger markers
            x_coords = [coord[0] for coord in coordinates]
            y_coords = [coord[1] for coord in coordinates]
            plt.scatter(x_coords, y_coords, c='red', s=200, zorder=5)

            # Plot route segments between cities using A* paths
            if route_paths:
                for path in route_paths:
                    # Draw path segments
                    for i in range(len(path)-1):
                        start = path[i]
                        end = path[i+1]

                        # Only draw if it's a grid-aligned movement
                        dx = end[0] - start[0]
                        dy = end[1] - start[1]

                        if dx == 0 or dy == 0:  # Only draw grid-aligned segments
                            # Draw segment
                            plt.plot([start[0], end[0]], [start[1], end[1]], 
                                    'b-', linewidth=2, zorder=3)

                            # Add direction arrow at midpoint
                            mid_x = (start[0] + end[0]) / 2
                            mid_y = (start[1] + end[1]) / 2
                            if dx == 0:  # Vertical movement
                                plt.arrow(mid_x, mid_y - dy*0.2,
                                        0, dy*0.4,
                                        head_width=0.2, head_length=0.3,
                                        fc='blue', ec='blue',
                                        length_includes_head=True,
                                        zorder=4)
                            else:  # Horizontal movement
                                plt.arrow(mid_x - dx*0.2, mid_y,
                                        dx*0.4, 0,
                                        head_width=0.2, head_length=0.3,
                                        fc='blue', ec='blue',
                                        length_includes_head=True,
                                        zorder=4)

            # Add city labels with larger font
            for i, (x, y) in enumerate(coordinates):
                plt.annotate(f'City {i}', (x, y), 
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=12, fontweight='bold')

            plt.title('Optimized Route with Obstacles', fontsize=14, pad=20)
            plt.xlabel('X Coordinate')
            plt.ylabel('Y Coordinate')
            plt.grid(True, alpha=0.3)
            plt.axis('equal')
            plt.xlim(-1, grid_size+1)
            plt.ylim(-1, grid_size+1)

            # Save with high DPI for better quality
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