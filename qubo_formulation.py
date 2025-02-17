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
                          penalty: float = 2.0) -> np.ndarray:
        """Create QUBO matrix with strong interactions."""
        if demands is None:
            demands = [1.0] * self.n_cities
            demands[depot_index] = 0.0

        size = self.n_cities * self.n_cities * self.n_vehicles
        Q = np.zeros((size, size))

        def get_index(i: int, j: int, k: int) -> int:
            """Generate unique index for each combination."""
            return k * self.n_cities * self.n_cities + i * self.n_cities + j

        # Scale distances to be between 0 and 1
        max_distance = np.max(distance_matrix)
        if max_distance > 0:
            distance_matrix = distance_matrix / max_distance

        # Base penalty scaled with problem size
        base_penalty = penalty * self.n_cities

        # 1. Distance terms with stronger coupling
        for k in range(self.n_vehicles):
            for i in range(self.n_cities):
                for j in range(self.n_cities):
                    if i != j:
                        idx1 = get_index(i, j, k)
                        Q[idx1, idx1] += distance_matrix[i, j]

                        # Add coupling terms between adjacent cities
                        for l in range(self.n_cities):
                            if l != i and l != j:
                                idx2 = get_index(j, l, k)
                                coupling = distance_matrix[i, j] * distance_matrix[j, l] * 0.5
                                Q[idx1, idx2] += coupling
                                Q[idx2, idx1] += coupling  # Ensure symmetry

        # 2. Visit constraints with scaled penalty
        visit_penalty = base_penalty
        for j in range(self.n_cities):
            if j != depot_index:
                for k1 in range(self.n_vehicles):
                    idx1 = get_index(depot_index, j, k1)
                    Q[idx1, idx1] += visit_penalty

                    # Add coupling between vehicles
                    for k2 in range(k1 + 1, self.n_vehicles):
                        idx2 = get_index(depot_index, j, k2)
                        Q[idx1, idx2] += visit_penalty * 0.5
                        Q[idx2, idx1] += visit_penalty * 0.5

        # 3. Vehicle capacity constraints with increased penalty
        for k in range(self.n_vehicles):
            if sum(demands) > self.vehicle_capacity[k]:
                capacity_penalty = base_penalty * (sum(demands) / self.vehicle_capacity[k])
                for i in range(self.n_cities):
                    if i != depot_index:
                        idx1 = get_index(depot_index, i, k)
                        Q[idx1, idx1] += demands[i] * capacity_penalty

                        # Add capacity coupling terms
                        for j in range(i + 1, self.n_cities):
                            if j != depot_index:
                                idx2 = get_index(depot_index, j, k)
                                coupling = capacity_penalty * demands[i] * demands[j] * 0.25
                                Q[idx1, idx2] += coupling
                                Q[idx2, idx1] += coupling

        return Q

    def decode_solution(self, binary_solution: List[int]) -> List[List[int]]:
        """Decode binary solution to routes with improved robustness."""
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