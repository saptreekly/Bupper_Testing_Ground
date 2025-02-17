import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from typing import List, Tuple, Set, Optional
import logging
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors

logger = logging.getLogger(__name__)

class CircuitVisualizer:
    def __init__(self):
        """Initialize the circuit visualizer."""
        try:
            plt.style.use('default')
            plt.switch_backend('Agg')
            # Create a color cycle for multiple vehicles
            self.colors = list(mcolors.TABLEAU_COLORS.values())
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

    def plot_route(self, coordinates: List[Tuple[int, int]], routes: List[List[int]], 
                   grid_size: int = 16, obstacles: Optional[Set[Tuple[str, int, int]]] = None, 
                   route_paths: Optional[List[List[Tuple[int, int]]]] = None,
                   save_path: str = "route.png"):
        """Plot and save the optimized routes."""
        try:
            # Create figure with adjusted size to maintain aspect ratio
            fig = plt.figure(figsize=(12, 12))
            ax = plt.gca()

            # Set equal aspect ratio explicitly
            ax.set_aspect('equal', adjustable='box')

            # Draw grid
            for i in range(grid_size + 1):
                plt.axhline(y=i, color='gray', linestyle='-', alpha=0.3)
                plt.axvline(x=i, color='gray', linestyle='-', alpha=0.3)

            # Draw obstacles if present
            blocked_squares = set()
            if obstacles:
                for obs_type, x, y in obstacles:
                    if obs_type == 'h':
                        blocked_squares.add((x, y))
                        blocked_squares.add((x+1, y))
                    else:
                        blocked_squares.add((x, y))
                        blocked_squares.add((x, y+1))

                for x, y in blocked_squares:
                    rect = Rectangle((x-0.5, y-0.5), 1, 1, 
                                      facecolor='darkred', edgecolor='black',
                                      alpha=0.8, zorder=2)
                    ax.add_patch(rect)

            # Plot cities
            x_coords = [coord[0] for coord in coordinates]
            y_coords = [coord[1] for coord in coordinates]
            plt.scatter(x_coords, y_coords, c='blue', s=200, zorder=5)

            # Plot routes for each vehicle with different colors
            path_index = 0
            for vehicle_idx, route in enumerate(routes):
                color = self.colors[vehicle_idx % len(self.colors)]

                # Get paths for this vehicle's route
                vehicle_paths = route_paths[path_index:path_index + len(route)-1]
                path_index += len(route)-1

                # Plot each path segment
                for path in vehicle_paths:
                    for i in range(len(path)-1):
                        start = path[i]
                        end = path[i+1]

                        dx = end[0] - start[0]
                        dy = end[1] - start[1]

                        if dx == 0 or dy == 0:  # Only draw grid-aligned segments
                            plt.plot([start[0], end[0]], [start[1], end[1]], 
                                       color=color, linestyle='-', linewidth=3, zorder=4,
                                       label=f'Vehicle {vehicle_idx}' if i == 0 else "")

                            # Add direction arrows
                            mid_x = (start[0] + end[0]) / 2
                            mid_y = (start[1] + end[1]) / 2
                            if dx == 0:  # Vertical movement
                                plt.arrow(mid_x, mid_y - dy*0.2,
                                          0, dy*0.4,
                                          head_width=0.3, head_length=0.4,
                                          fc=color, ec=color,
                                          length_includes_head=True,
                                          zorder=6)
                            else:  # Horizontal movement
                                plt.arrow(mid_x - dx*0.2, mid_y,
                                          dx*0.4, 0,
                                          head_width=0.3, head_length=0.4,
                                          fc=color, ec=color,
                                          length_includes_head=True,
                                          zorder=6)

            # Add city labels
            for i, (x, y) in enumerate(coordinates):
                label = 'Depot' if i == 0 else f'City {i}'
                plt.annotate(label, (x, y), 
                            xytext=(5, 5), textcoords='offset points',
                            fontsize=12, fontweight='bold')

            plt.title('Optimized Vehicle Routes', fontsize=14, pad=20)
            plt.xlabel('X Coordinate')
            plt.ylabel('Y Coordinate')
            plt.grid(True, alpha=0.3)
            plt.axis('equal')
            plt.xlim(-1, grid_size+1)
            plt.ylim(-1, grid_size+1)

            # Add legend for vehicles
            plt.legend()

            # Save with tight layout and dpi adjustment
            plt.savefig(save_path, bbox_inches='tight', dpi=300, 
                       pad_inches=0.5)  # Add padding to prevent cutoff
            plt.close()
            logger.info(f"Route plot saved to {save_path}")

        except Exception as e:
            logger.error(f"Error plotting route: {str(e)}")
            raise

    def plot_circuit_results(self, measurements: List[float], title: str = "QAOA Results", save_path: str = "circuit_results.png"):
        """Plot and save measurement results from the quantum circuit."""
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