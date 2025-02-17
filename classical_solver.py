import itertools
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
import time

logger = logging.getLogger(__name__)

def solve_tsp_brute_force(distance_matrix: np.ndarray, depot_index: int = 0) -> Tuple[List[int], float]:
    """
    Solve Traveling Salesman Problem using brute force approach.

    Args:
        distance_matrix: Matrix of distances between cities
        depot_index: Index of the depot (starting/ending point)

    Returns:
        Tuple containing optimal route and its length
    """
    n_cities = len(distance_matrix)

    # Generate all possible city permutations (excluding depot)
    other_cities = [i for i in range(n_cities) if i != depot_index]
    min_length = float('inf')
    best_route = None

    # Try all possible permutations
    for perm in itertools.permutations(other_cities):
        # Add depot at start and end
        route = [depot_index] + list(perm) + [depot_index]

        # Calculate route length
        length = 0
        for i in range(len(route) - 1):
            length += distance_matrix[route[i], route[i+1]]

        # Update if better route found
        if length < min_length:
            min_length = length
            best_route = route
            logger.info(f"New best route found: {route} with length {length}")

    return best_route, min_length

def clarke_wright_savings(distance_matrix: np.ndarray, 
                         demands: Optional[List[float]] = None,
                         depot_index: int = 0,
                         capacity: float = float('inf')) -> Tuple[List[List[int]], float]:
    """
    Solve Vehicle Routing Problem using Clarke-Wright Savings Algorithm.
    This is an industry-standard heuristic used by many logistics companies.

    Args:
        distance_matrix: Matrix of distances between cities
        demands: List of demands for each city (optional)
        depot_index: Index of the depot
        capacity: Vehicle capacity constraint

    Returns:
        Tuple containing list of routes and total distance
    """
    try:
        n_cities = len(distance_matrix)
        if demands is None:
            demands = [1.0] * n_cities
            demands[depot_index] = 0.0

        # Calculate savings for all pairs
        savings = {}
        for i in range(n_cities):
            for j in range(i + 1, n_cities):
                if i != depot_index and j != depot_index:
                    # Classical savings formula: cost(0,i) + cost(0,j) - cost(i,j)
                    saving = (distance_matrix[depot_index, i] + 
                            distance_matrix[depot_index, j] - 
                            distance_matrix[i, j])
                    savings[(i, j)] = saving

        # Sort savings in descending order
        sorted_savings = sorted(savings.items(), key=lambda x: x[1], reverse=True)

        # Initialize routes: start with direct depot-customer-depot routes
        routes = [[depot_index, i, depot_index] for i in range(n_cities) if i != depot_index]
        route_demands = [demands[i] for i in range(n_cities) if i != depot_index]

        # Merge routes based on savings
        for (i, j), saving in sorted_savings:
            # Find routes containing i and j
            route_i = None
            route_j = None
            for idx, route in enumerate(routes):
                if i in route:
                    route_i = idx
                if j in route:
                    route_j = idx

            if route_i is None or route_j is None or route_i == route_j:
                continue

            # Check if merge is feasible (endpoints and capacity)
            if (route_demands[route_i] + route_demands[route_j] <= capacity and
                (routes[route_i].index(i) in [1, len(routes[route_i])-2]) and
                (routes[route_j].index(j) in [1, len(routes[route_j])-2])):

                # Merge routes
                new_route = merge_routes(routes[route_i], routes[route_j], i, j)
                new_demand = route_demands[route_i] + route_demands[route_j]

                # Update routes list
                routes.pop(max(route_i, route_j))
                routes.pop(min(route_i, route_j))
                route_demands.pop(max(route_i, route_j))
                route_demands.pop(min(route_i, route_j))

                routes.append(new_route)
                route_demands.append(new_demand)

        # Calculate total distance
        total_distance = 0
        for route in routes:
            for i in range(len(route) - 1):
                total_distance += distance_matrix[route[i], route[i+1]]

        logger.info(f"Clarke-Wright solution found with {len(routes)} routes")
        for idx, route in enumerate(routes):
            logger.info(f"Route {idx}: {route} (demand: {route_demands[idx]:.2f})")
        logger.info(f"Total distance: {total_distance}")

        return routes, total_distance

    except Exception as e:
        logger.error(f"Error in Clarke-Wright algorithm: {str(e)}")
        raise

def merge_routes(route1: List[int], route2: List[int], i: int, j: int) -> List[int]:
    """Helper function to merge two routes based on savings."""
    # Remove depot from end of first route and start of second route
    if route1.index(i) == 1:
        route1 = route1[::-1]
    if route2.index(j) != 1:
        route2 = route2[::-1]

    # Merge routes
    merged = route1[:-1] + route2[1:]
    return merged

def compare_solutions(distance_matrix: np.ndarray, 
                     quantum_solution: List[int],
                     classical_solution: List[int],
                     quantum_time: float,
                     classical_time: float) -> Dict:
    """
    Compare quantum and classical solutions.

    Returns:
        Dictionary containing comparison metrics
    """
    quantum_distance = sum(distance_matrix[quantum_solution[i], quantum_solution[i+1]] 
                         for i in range(len(quantum_solution)-1))
    classical_distance = sum(distance_matrix[classical_solution[i], classical_solution[i+1]] 
                          for i in range(len(classical_solution)-1))

    return {
        'quantum_distance': quantum_distance,
        'classical_distance': classical_distance,
        'quantum_time': quantum_time,
        'classical_time': classical_time,
        'ratio': quantum_distance / classical_distance if classical_distance > 0 else float('inf'),
        'speedup': classical_time / quantum_time if quantum_time > 0 else float('inf')
    }