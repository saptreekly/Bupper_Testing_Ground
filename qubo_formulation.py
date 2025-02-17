import numpy as np
from typing import List, Dict, Tuple, Optional

class QUBOFormulation:
    def __init__(self, n_cities: int, n_vehicles: int, vehicle_capacity: Optional[List[float]] = None):
        """Initialize QUBO formulation with problem size limits."""
        self.n_cities = min(n_cities, 5)  # Limit to maximum 5 cities
        self.n_vehicles = n_vehicles
        self.vehicle_capacity = vehicle_capacity if vehicle_capacity else [float('inf')] * n_vehicles

    def create_distance_matrix(self, coordinates: List[Tuple[float, float]]) -> np.ndarray:
        """Create and normalize distance matrix."""
        n = len(coordinates)
        distance_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    x1, y1 = coordinates[i]
                    x2, y2 = coordinates[j]
                    distance_matrix[i, j] = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        return distance_matrix

    def create_qubo_matrix(self, distance_matrix: np.ndarray, 
                          demands: Optional[List[float]] = None,
                          depot_index: int = 0,
                          penalty: float = 0.05) -> np.ndarray:  # Further reduced penalty
        """Simplified QUBO matrix creation with minimal terms."""
        if demands is None:
            demands = [1.0] * self.n_cities
            demands[depot_index] = 0.0

        size = self.n_cities * self.n_cities * self.n_vehicles
        Q = np.zeros((size, size))

        def get_index(i: int, j: int, k: int) -> int:
            return k * self.n_cities * self.n_cities + i * self.n_cities + j

        # Scale distance matrix
        max_distance = np.max(distance_matrix)
        if max_distance > 0:
            distance_matrix = distance_matrix / max_distance

        # Only essential terms with minimal penalties
        # 1. Distance terms (minimal scaling)
        for k in range(self.n_vehicles):
            for i in range(self.n_cities):
                for j in range(self.n_cities):
                    if i != j and distance_matrix[i, j] > 0:
                        idx = get_index(i, j, k)
                        Q[idx, idx] += distance_matrix[i, j] * 0.1

        # 2. Visit constraints (reduced penalty)
        for j in range(self.n_cities):
            if j != depot_index:
                for k in range(self.n_vehicles):
                    for i in range(self.n_cities):
                        if i != j:
                            idx = get_index(i, j, k)
                            Q[idx, idx] += penalty

        # 3. Vehicle constraints (minimal terms)
        for k in range(self.n_vehicles):
            # Start at depot
            for j in range(self.n_cities):
                if j != depot_index:
                    idx = get_index(depot_index, j, k)
                    Q[idx, idx] -= penalty * 0.1

            # End at depot
            for i in range(self.n_cities):
                if i != depot_index:
                    idx = get_index(i, depot_index, k)
                    Q[idx, idx] -= penalty * 0.1

        return Q

    def decode_solution(self, binary_solution: List[int]) -> List[List[int]]:
        """Simplified solution decoding."""
        solution_matrix = np.array(binary_solution).reshape(self.n_vehicles, 
                                                          self.n_cities, 
                                                          self.n_cities)
        routes = []

        for k in range(self.n_vehicles):
            route = [0]  # Start at depot
            current = 0
            visited = {0}

            while len(visited) < self.n_cities and len(route) < self.n_cities + 1:
                next_city = -1
                max_prob = -1

                for j in range(self.n_cities):
                    if j not in visited and solution_matrix[k, current, j] > max_prob:
                        max_prob = solution_matrix[k, current, j]
                        next_city = j

                if next_city == -1:
                    break

                route.append(next_city)
                visited.add(next_city)
                current = next_city

            route.append(0)  # Return to depot
            routes.append(route)

        return routes