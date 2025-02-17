import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from typing import List, Tuple, Set, Optional, Dict, Any
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

                # Get paths for this vehicle's route if available
                vehicle_paths = []
                if route_paths:
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

    def plot_solution_comparison(self, coordinates: List[Tuple[float, float]], 
                               quantum_route: List[int], classical_route: List[int],
                               quantum_metrics: Dict, classical_metrics: Dict,
                               save_path: str = "solution_comparison.png"):
        """Plot quantum vs classical solution comparison with geographic coordinates."""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

            # Plot quantum solution
            ax1.set_title('Quantum Solution\n'
                         f'Distance: {quantum_metrics["distance"]:.2f}\n'
                         f'Time: {quantum_metrics["time"]:.2f}s')
            self._plot_route_on_axis(ax1, coordinates, [quantum_route])

            # Plot classical solution
            ax2.set_title('Classical Solution\n'
                         f'Distance: {classical_metrics["distance"]:.2f}\n'
                         f'Time: {classical_metrics["time"]:.2f}s')
            self._plot_route_on_axis(ax2, coordinates, [classical_route])

            # Add comparison metrics
            plt.figtext(0.5, 0.02, 
                       f'Approximation Ratio: {quantum_metrics["distance"]/classical_metrics["distance"]:.3f}\n'
                       f'Quantum/Classical Time Ratio: {quantum_metrics["time"]/classical_metrics["time"]:.3f}',
                       ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

            plt.tight_layout()
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
            logger.info(f"Comparison plot saved to {save_path}")

        except Exception as e:
            logger.error(f"Error plotting solution comparison: {str(e)}")
            raise

    def _plot_route_on_axis(self, ax, coordinates: List[Tuple[float, float]], 
                           routes: List[List[int]]):
        """Helper function to plot a route on a given axis using geographic coordinates."""
        # Plot cities
        lats = [coord[0] for coord in coordinates]
        lons = [coord[1] for coord in coordinates]
        ax.scatter(lons, lats, c='blue', s=200, zorder=5)

        # Plot routes
        for route_idx, route in enumerate(routes):
            color = self.colors[route_idx % len(self.colors)]
            for i in range(len(route)-1):
                start = coordinates[route[i]]
                end = coordinates[route[i+1]]
                ax.plot([start[1], end[1]], [start[0], end[0]], 
                       color=color, linestyle='-', linewidth=2,
                       label=f'Route {route_idx}' if i == 0 else "")

                # Add direction arrows
                mid_lat = (start[0] + end[0]) / 2
                mid_lon = (start[1] + end[1]) / 2
                dlat = end[0] - start[0]
                dlon = end[1] - start[1]
                ax.arrow(mid_lon - dlon*0.1, mid_lat - dlat*0.1,
                        dlon*0.2, dlat*0.2,
                        head_width=0.001, head_length=0.002,
                        fc=color, ec=color,
                        length_includes_head=True,
                        zorder=6)

        # Add city labels
        for i, (lat, lon) in enumerate(coordinates):
            label = 'Depot' if i == 0 else f'City {i}'
            ax.annotate(label, (lon, lat), 
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=10)

        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

    def plot_benchmark_results(self, metrics_list: List[Dict], save_path: str = "benchmark_results.png"):
        """Plot and save comprehensive benchmark results."""
        try:
            fig = plt.figure(figsize=(15, 10))
            gs = plt.GridSpec(2, 2)

            # Plot 1: Optimization times comparison
            ax1 = fig.add_subplot(gs[0, 0])
            backends = sorted(set(m['backend'] for m in metrics_list))
            problem_sizes = sorted(set(m['problem_size']['n_cities'] for m in metrics_list))

            for backend in backends:
                times = [next(m['optimization_time'] for m in metrics_list 
                            if m['backend'] == backend and 
                            m['problem_size']['n_cities'] == size)
                        for size in problem_sizes]
                ax1.plot(problem_sizes, times, 'o-', label=f'{backend}')

            ax1.set_xlabel('Number of Cities')
            ax1.set_ylabel('Optimization Time (s)')
            ax1.set_title('Optimization Time vs Problem Size')
            ax1.legend()
            ax1.grid(True)

            # Plot 2: Solution quality comparison
            ax2 = fig.add_subplot(gs[0, 1])
            for backend in backends:
                gaps = [next(m['quantum_classical_gap'] for m in metrics_list 
                           if m['backend'] == backend and 
                           m['problem_size']['n_cities'] == size)
                       for size in problem_sizes]
                ax2.plot(problem_sizes, gaps, 'o-', label=f'{backend}')

            ax2.set_xlabel('Number of Cities')
            ax2.set_ylabel('Gap to Classical Solution')
            ax2.set_title('Solution Quality vs Problem Size')
            ax2.legend()
            ax2.grid(True)

            # Plot 3: Convergence history
            ax3 = fig.add_subplot(gs[1, :])
            for metrics in metrics_list:
                if 'convergence_history' in metrics:
                    history = metrics['convergence_history']
                    label = f"{metrics['backend']} ({metrics['problem_size']['n_cities']} cities)"
                    ax3.plot(history, label=label, alpha=0.7)

            ax3.set_xlabel('Iteration')
            ax3.set_ylabel('Cost')
            ax3.set_title('Convergence History')
            ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax3.grid(True)

            plt.tight_layout()
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
            logger.info(f"Benchmark results plot saved to {save_path}")

        except Exception as e:
            logger.error(f"Error plotting benchmark results: {str(e)}")
            raise

    def plot_performance_metrics(self, metrics: Dict[str, Any], save_path: str = "performance_metrics.png"):
        """Plot detailed performance metrics from a single run."""
        try:
            fig = plt.figure(figsize=(12, 8))
            gs = plt.GridSpec(2, 2)

            # Plot 1: Time breakdown
            ax1 = fig.add_subplot(gs[0, 0])
            time_metrics = {k: v for k, v in metrics.items() if 'time' in k and isinstance(v, (int, float))}
            ax1.pie(time_metrics.values(), labels=time_metrics.keys(), autopct='%1.1f%%')
            ax1.set_title('Execution Time Breakdown')

            # Plot 2: Solution characteristics
            ax2 = fig.add_subplot(gs[0, 1])
            if 'solution_length' in metrics and 'classical_solution_time' in metrics:
                values = [metrics['solution_length'], metrics.get('classical_solution_time', 0)]
                labels = ['Quantum', 'Classical']
                ax2.bar(labels, values)
                ax2.set_title('Solution Quality Comparison')
                ax2.set_ylabel('Route Length')

            # Plot 3: Convergence analysis
            ax3 = fig.add_subplot(gs[1, :])
            if 'convergence_history' in metrics:
                history = metrics['convergence_history']
                ax3.plot(history, 'b-', label='Cost')
                ax3.set_xlabel('Iteration')
                ax3.set_ylabel('Cost')
                ax3.set_title('Optimization Convergence')
                ax3.grid(True)

            plt.tight_layout()
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
            logger.info(f"Performance metrics plot saved to {save_path}")

        except Exception as e:
            logger.error(f"Error plotting performance metrics: {str(e)}")
            raise
    
    def plot_time_series_analysis(self, metrics_list: List[Dict], save_path: str = "time_series_analysis.png"):
        """Plot detailed time series analysis of optimization runs."""
        try:
            fig = plt.figure(figsize=(15, 8))
            gs = plt.GridSpec(2, 1, height_ratios=[2, 1])

            # Plot 1: Convergence curves with confidence bands
            ax1 = fig.add_subplot(gs[0])
            backends = sorted(set(m['backend'] for m in metrics_list))
            problem_sizes = sorted(set(m['problem_size']['n_cities'] for m in metrics_list))

            for backend in backends:
                # Collect all convergence histories for this backend
                pure_histories = [m['convergence_history'] for m in metrics_list 
                               if m['backend'] == backend and not m.get('hybrid', False)]
                hybrid_histories = [m['convergence_history'] for m in metrics_list 
                                 if m['backend'] == backend and m.get('hybrid', True)]

                # Plot pure quantum convergence
                if pure_histories:
                    max_len = max(len(h) for h in pure_histories)
                    normalized = np.array([np.interp(np.linspace(0, 1, max_len),
                                                  np.linspace(0, 1, len(h)), h)
                                        for h in pure_histories])
                    mean_curve = np.mean(normalized, axis=0)
                    std_curve = np.std(normalized, axis=0)

                    x = range(max_len)
                    ax1.plot(x, mean_curve, label=f'{backend} pure', linewidth=2)
                    ax1.fill_between(x, mean_curve - std_curve, mean_curve + std_curve,
                                  alpha=0.2)

                # Plot hybrid convergence
                if hybrid_histories:
                    max_len = max(len(h) for h in hybrid_histories)
                    normalized = np.array([np.interp(np.linspace(0, 1, max_len),
                                                  np.linspace(0, 1, len(h)), h)
                                        for h in hybrid_histories])
                    mean_curve = np.mean(normalized, axis=0)
                    std_curve = np.std(normalized, axis=0)

                    x = range(max_len)
                    ax1.plot(x, mean_curve, '--', label=f'{backend} hybrid', linewidth=2)
                    ax1.fill_between(x, mean_curve - std_curve, mean_curve + std_curve,
                                  alpha=0.2)

            ax1.set_xlabel('Normalized Iteration')
            ax1.set_ylabel('Cost')
            ax1.set_title('Convergence Analysis with Uncertainty')
            ax1.legend()
            ax1.grid(True)

            # Plot 2: Performance scaling
            ax2 = fig.add_subplot(gs[1])
            for backend in backends:
                for is_hybrid in [False, True]:
                    scaling_data = []
                    for size in problem_sizes:
                        metrics = [m for m in metrics_list 
                                if m['backend'] == backend and 
                                m['problem_size']['n_cities'] == size and
                                m.get('hybrid', False) == is_hybrid]
                        if metrics:
                            mean_time = np.mean([m['total_time'] for m in metrics])
                            scaling_data.append(mean_time)

                    if scaling_data:
                        label = f'{backend} {"hybrid" if is_hybrid else "pure"}'
                        style = '--' if is_hybrid else '-'
                        ax2.plot(problem_sizes[:len(scaling_data)], scaling_data, 
                               f'o{style}', label=label)

            ax2.set_xlabel('Number of Cities')
            ax2.set_ylabel('Total Time (s)')
            ax2.set_title('Performance Scaling')
            ax2.legend()
            ax2.grid(True)

            plt.tight_layout()
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
            logger.info(f"Time series analysis plot saved to {save_path}")

        except Exception as e:
            logger.error(f"Error plotting time series analysis: {str(e)}")
            raise

    def plot_cross_validation_metrics(self, metrics_list: List[Dict], save_path: str = "cross_validation_metrics.png"):
        """Plot cross-validation metrics comparing hybrid and pure approaches."""
        try:
            fig = plt.figure(figsize=(15, 10))
            gs = plt.GridSpec(2, 2)

            # Plot 1: Solution quality distribution
            ax1 = fig.add_subplot(gs[0, 0])
            backends = sorted(set(m['backend'] for m in metrics_list))

            data = []
            labels = []
            for backend in backends:
                for is_hybrid in [False, True]:
                    metrics = [m for m in metrics_list 
                             if m['backend'] == backend and 
                             m.get('hybrid', False) == is_hybrid]
                    if metrics:
                        gaps = [m['quantum_classical_gap'] for m in metrics]
                        data.append(gaps)
                        labels.append(f"{backend}\n{'hybrid' if is_hybrid else 'pure'}")

            ax1.boxplot(data, labels=labels)
            ax1.set_ylabel('Gap to Classical Solution')
            ax1.set_title('Solution Quality Distribution')
            plt.setp(ax1.get_xticklabels(), rotation=45)
            ax1.grid(True)

            # Plot 2: Time efficiency comparison
            ax2 = fig.add_subplot(gs[0, 1])
            for backend in backends:
                pure_metrics = [m for m in metrics_list 
                              if m['backend'] == backend and 
                              not m.get('hybrid', False)]
                hybrid_metrics = [m for m in metrics_list 
                                if m['backend'] == backend and 
                                m.get('hybrid', True)]

                if pure_metrics and hybrid_metrics:
                    sizes = sorted(set(m['problem_size']['n_cities'] for m in pure_metrics))
                    pure_times = []
                    hybrid_times = []

                    for size in sizes:
                        pure_time = np.mean([m['total_time'] for m in pure_metrics 
                                          if m['problem_size']['n_cities'] == size])
                        hybrid_time = np.mean([m['total_time'] for m in hybrid_metrics 
                                            if m['problem_size']['n_cities'] == size])
                        pure_times.append(pure_time)
                        hybrid_times.append(hybrid_time)

                    ax2.plot(sizes, [h/p for h, p in zip(hybrid_times, pure_times)],
                            'o-', label=backend)

            ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
            ax2.set_xlabel('Number of Cities')
            ax2.set_ylabel('Hybrid/Pure Time Ratio')
            ax2.set_title('Time Efficiency Comparison')
            ax2.legend()
            ax2.grid(True)

            # Plot 3: Convergence stability
            ax3 = fig.add_subplot(gs[1, :])
            for backend in backends:
                for is_hybrid in [False, True]:
                    metrics = [m for m in metrics_list 
                             if m['backend'] == backend and 
                             m.get('hybrid', False) == is_hybrid]
                    if metrics:
                        variances = [m['cost_variance'] for m in metrics if 'cost_variance' in m]
                        sizes = [m['problem_size']['n_cities'] for m in metrics if 'cost_variance' in m]

                        if variances and sizes:
                            style = '--' if is_hybrid else '-'
                            label = f"{backend} {'hybrid' if is_hybrid else 'pure'}"
                            ax3.plot(sizes, variances, f'o{style}', label=label)

            ax3.set_xlabel('Number of Cities')
            ax3.set_ylabel('Cost Variance')
            ax3.set_title('Convergence Stability Analysis')
            ax3.legend()
            ax3.grid(True)

            plt.tight_layout()
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
            logger.info(f"Cross-validation metrics plot saved to {save_path}")

        except Exception as e:
            logger.error(f"Error plotting cross-validation metrics: {str(e)}")
            raise