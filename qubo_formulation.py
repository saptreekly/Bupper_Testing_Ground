import numpy as np
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class QUBOFormulation:
    def __init__(self, n_cities: int, n_vehicles: int, vehicle_capacity: List[float], backend: str = 'pennylane'):
        """Initialize QUBO problem for vehicle routing."""
        self.n_cities = n_cities
        self.n_vehicles = n_vehicles
        self.vehicle_capacity = vehicle_capacity
        self.backend = backend
        
        # Validate inputs
        if n_cities < 2:
            raise ValueError("Number of cities must be at least 2")
        if n_vehicles < 1:
            raise ValueError("Number of vehicles must be at least 1")
        if len(vehicle_capacity) != n_vehicles:
            raise ValueError("Vehicle capacity list must match number of vehicles")

        logger.info(f"Initialized QUBO formulation with {n_cities} cities and {n_vehicles} vehicles")

    def create_distance_matrix(self, coordinates: List[tuple]) -> np.ndarray:
        """Create distance matrix from coordinates."""
        if len(coordinates) != self.n_cities:
            raise ValueError(f"Expected {self.n_cities} coordinates, got {len(coordinates)}")
        
        distance_matrix = np.zeros((self.n_cities, self.n_cities))
        for i in range(self.n_cities):
            for j in range(self.n_cities):
                if i != j:
                    distance_matrix[i,j] = self._calculate_distance(coordinates[i], coordinates[j])
        
        return distance_matrix

    def create_qubo_matrix(self, distance_matrix: np.ndarray, demands: List[float] = None, 
                          penalty: float = 1.0) -> np.ndarray:
        """Create QUBO matrix for the vehicle routing problem."""
        if demands is None:
            demands = [0.0] + [1.0] * (self.n_cities - 1)  # First city (depot) has no demand
        
        n_qubits = self.n_cities * self.n_cities * self.n_vehicles
        Q = np.zeros((n_qubits, n_qubits))
        
        # Add distance terms
        for v in range(self.n_vehicles):
            for i in range(self.n_cities):
                for j in range(self.n_cities):
                    if i != j:
                        idx1 = self._get_qubit_index(i, j, v)
                        Q[idx1, idx1] = distance_matrix[i,j]
        
        # Add constraint terms with penalty
        self._add_constraint_terms(Q, penalty, demands)
        
        return Q

    def decode_solution(self, binary_solution: List[int]) -> List[List[int]]:
        """Decode binary solution into vehicle routes."""
        routes = []
        solution_matrix = np.array(binary_solution).reshape(
            (self.n_vehicles, self.n_cities, self.n_cities))
        
        for v in range(self.n_vehicles):
            route = [0]  # Start at depot
            current = 0
            visited = {0}
            
            while len(visited) < self.n_cities:
                next_city = -1
                for j in range(self.n_cities):
                    if j not in visited and solution_matrix[v, current, j] == 1:
                        next_city = j
                        break
                
                if next_city == -1:
                    break
                    
                route.append(next_city)
                visited.add(next_city)
                current = next_city
            
            if len(route) > 1:
                route.append(0)  # Return to depot
                routes.append(route)
        
        return routes

    def _calculate_distance(self, coord1: tuple, coord2: tuple) -> float:
        """Calculate Euclidean distance between two coordinates."""
        return np.sqrt(sum((c1 - c2) ** 2 for c1, c2 in zip(coord1, coord2)))

    def _get_qubit_index(self, i: int, j: int, v: int) -> int:
        """Convert city indices and vehicle index to qubit index."""
        return v * (self.n_cities * self.n_cities) + i * self.n_cities + j

    def _add_constraint_terms(self, Q: np.ndarray, penalty: float, demands: List[float]):
        """Add constraint terms to QUBO matrix."""
        # Each city must be visited exactly once
        for j in range(1, self.n_cities):  # Skip depot
            for v1 in range(self.n_vehicles):
                for v2 in range(self.n_vehicles):
                    for i1 in range(self.n_cities):
                        for i2 in range(self.n_cities):
                            if i1 != j and i2 != j:
                                idx1 = self._get_qubit_index(i1, j, v1)
                                idx2 = self._get_qubit_index(i2, j, v2)
                                if idx1 == idx2:
                                    Q[idx1, idx1] += penalty
                                else:
                                    Q[idx1, idx2] += penalty
        
        # Vehicle capacity constraints
        for v in range(self.n_vehicles):
            capacity = self.vehicle_capacity[v]
            for i in range(self.n_cities):
                for j in range(1, self.n_cities):  # Skip depot
                    idx = self._get_qubit_index(i, j, v)
                    Q[idx, idx] += penalty * max(0, demands[j] - capacity)
