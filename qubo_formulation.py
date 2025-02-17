import numpy as np
from typing import List, Dict, Tuple, Optional

class QUBOFormulation:
    def __init__(self, n_cities: int, n_vehicles: int, vehicle_capacity: Optional[List[float]] = None):
        """
        Initialize QUBO formulation for vehicle routing problem.

        Args:
            n_cities (int): Number of cities in the routing problem
            n_vehicles (int): Number of available vehicles
            vehicle_capacity (Optional[List[float]]): Capacity constraints for each vehicle
        """
        self.n_cities = n_cities
        self.n_vehicles = n_vehicles
        self.vehicle_capacity = vehicle_capacity if vehicle_capacity else [float('inf')] * n_vehicles

    def create_distance_matrix(self, coordinates: List[Tuple[float, float]]) -> np.ndarray:
        """
        Create distance matrix from city coordinates.

        Args:
            coordinates (List[Tuple[float, float]]): List of (x, y) coordinates

        Returns:
            np.ndarray: Distance matrix
        """
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
        """
        Create QUBO matrix for the Vehicle Routing Problem.

        Args:
            distance_matrix (np.ndarray): Matrix of distances between cities
            demands (Optional[List[float]]): Demand at each city
            depot_index (int): Index of the depot location
            penalty (float): Penalty coefficient for constraints

        Returns:
            np.ndarray: QUBO matrix
        """
        if demands is None:
            demands = [1.0] * self.n_cities
            demands[depot_index] = 0.0

        # Size of the binary variable array: n_cities * n_cities * n_vehicles
        size = self.n_cities * self.n_cities * self.n_vehicles
        Q = np.zeros((size, size))

        # Helper function to get index in the QUBO matrix
        def get_index(i: int, j: int, k: int) -> int:
            return k * self.n_cities * self.n_cities + i * self.n_cities + j

        # Add distance terms
        for k in range(self.n_vehicles):
            for i in range(self.n_cities):
                for j in range(self.n_cities):
                    if i != j:
                        idx = get_index(i, j, k)
                        Q[idx, idx] += distance_matrix[i, j]

        # Constraint 1: Each city must be visited exactly once by any vehicle
        for j in range(self.n_cities):
            if j != depot_index:
                for k1 in range(self.n_vehicles):
                    for k2 in range(k1, self.n_vehicles):
                        for i1 in range(self.n_cities):
                            for i2 in range(self.n_cities):
                                if i1 != j and i2 != j:
                                    idx1 = get_index(i1, j, k1)
                                    idx2 = get_index(i2, j, k2)
                                    if k1 == k2 and i1 == i2:
                                        continue
                                    Q[idx1, idx2] += penalty

        # Constraint 2: Vehicle capacity constraints
        for k in range(self.n_vehicles):
            for i in range(self.n_cities):
                for j in range(self.n_cities):
                    if i != j:
                        idx = get_index(i, j, k)
                        capacity_violation = max(0, demands[j] - self.vehicle_capacity[k])
                        Q[idx, idx] += penalty * capacity_violation

        # Constraint 3: Flow conservation - enter and exit each city
        for k in range(self.n_vehicles):
            for j in range(self.n_cities):
                if j != depot_index:
                    # Sum of incoming = Sum of outgoing = 1
                    for i1 in range(self.n_cities):
                        for i2 in range(self.n_cities):
                            if i1 != j and i2 != j:
                                idx1 = get_index(i1, j, k)
                                idx2 = get_index(j, i2, k)
                                Q[idx1, idx2] += penalty

        # Constraint 4: All vehicles must start and end at depot
        for k in range(self.n_vehicles):
            # Start at depot
            for j in range(self.n_cities):
                if j != depot_index:
                    idx = get_index(depot_index, j, k)
                    Q[idx, idx] -= penalty/2

            # End at depot
            for i in range(self.n_cities):
                if i != depot_index:
                    idx = get_index(i, depot_index, k)
                    Q[idx, idx] -= penalty/2

        # Make the matrix symmetric
        Q = (Q + Q.T) / 2

        return Q

    def decode_solution(self, binary_solution: List[int]) -> List[List[int]]:
        """
        Decode binary solution to vehicle routes.

        Args:
            binary_solution (List[int]): Binary solution from QAOA

        Returns:
            List[List[int]]: List of routes, one for each vehicle
        """
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