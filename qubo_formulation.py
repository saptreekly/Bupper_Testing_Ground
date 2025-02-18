import numpy as np
from typing import List, Dict, Any, Tuple
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
                        penalty: float = 2.0) -> Tuple[np.ndarray, List[Tuple[float, Tuple[int, int]]]]:
        """Create QUBO matrix for the vehicle routing problem."""
        if demands is None:
            demands = [0.0] + [1.0] * (self.n_cities - 1)  # First city (depot) has no demand

        n_qubits = self.n_cities * self.n_cities * self.n_vehicles
        Q = np.zeros((n_qubits, n_qubits))
        cost_terms = []

        # Normalize distance matrix to [0, 1] range
        max_distance = np.max(distance_matrix)
        if max_distance > 0:
            normalized_distances = distance_matrix / max_distance
        else:
            normalized_distances = distance_matrix

        logger.info(f"Creating QUBO matrix with {n_qubits} qubits")
        logger.info(f"Maximum distance: {max_distance}, using normalized distances")
        logger.info(f"Distance matrix shape: {distance_matrix.shape}")

        # Add distance terms with normalized coefficients
        for v in range(self.n_vehicles):
            for i in range(self.n_cities):
                for j in range(self.n_cities):
                    if i != j:
                        idx1 = self._get_qubit_index(i, j, v)
                        coeff = float(normalized_distances[i,j])
                        Q[idx1, idx1] = coeff
                        cost_terms.append((coeff, (idx1, idx1)))  # Add diagonal terms

                        # Add interaction terms between consecutive cities
                        for k in range(self.n_cities):
                            if k != i and k != j:
                                idx2 = self._get_qubit_index(j, k, v)
                                interaction_coeff = float(normalized_distances[i,j] + normalized_distances[j,k])
                                Q[idx1, idx2] = interaction_coeff
                                if idx1 < idx2:  # Avoid duplicates
                                    cost_terms.append((interaction_coeff, (idx1, idx2)))

        # Add constraint terms with penalty
        self._add_constraint_terms(Q, penalty, demands)

        # Filter out near-zero terms
        threshold = 1e-8  # Very small threshold to keep most meaningful terms
        filtered_cost_terms = [(coeff, indices) for coeff, indices in cost_terms if abs(coeff) > threshold]

        logger.info(f"Generated {len(filtered_cost_terms)} cost terms from QUBO matrix")
        logger.info(f"Matrix statistics - Min: {np.min(Q):.6f}, Max: {np.max(Q):.6f}, Mean: {np.mean(np.abs(Q)):.6f}")

        return Q, filtered_cost_terms

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
        constraint_terms = 0

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
                                constraint_terms += 1

        # Vehicle capacity constraints with normalized demands
        max_demand = max(demands)
        normalized_demands = [d/max_demand if max_demand > 0 else d for d in demands]

        for v in range(self.n_vehicles):
            capacity = self.vehicle_capacity[v]
            for i in range(self.n_cities):
                for j in range(1, self.n_cities):  # Skip depot
                    idx = self._get_qubit_index(i, j, v)
                    penalty_term = penalty * max(0, normalized_demands[j] - capacity/max_demand if max_demand > 0 else 0)
                    Q[idx, idx] += penalty_term
                    constraint_terms += 1

        logger.info(f"Added {constraint_terms} constraint terms to QUBO matrix")