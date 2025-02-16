import numpy as np
from typing import List, Dict, Tuple

class QUBOFormulation:
    def __init__(self, n_cities: int):
        """
        Initialize QUBO formulation for routing problem.
        
        Args:
            n_cities (int): Number of cities in the routing problem
        """
        self.n_cities = n_cities
        
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
    
    def create_qubo_matrix(self, distance_matrix: np.ndarray, penalty: float = 1.0) -> np.ndarray:
        """
        Create QUBO matrix for the routing problem.
        
        Args:
            distance_matrix (np.ndarray): Matrix of distances between cities
            penalty (float): Penalty coefficient for constraints
        
        Returns:
            np.ndarray: QUBO matrix
        """
        n = self.n_cities
        size = n * n
        Q = np.zeros((size, size))
        
        # Add distance terms
        for i in range(n):
            for j in range(n):
                for k in range(n-1):
                    idx1 = i * n + k
                    idx2 = j * n + (k + 1)
                    Q[idx1, idx2] = distance_matrix[i, j]
        
        # Add constraint terms (one city per time step)
        for t in range(n):
            for i in range(n):
                for j in range(i+1, n):
                    idx1 = i * n + t
                    idx2 = j * n + t
                    Q[idx1, idx2] += penalty
        
        # Add constraint terms (each city visited once)
        for i in range(n):
            for t1 in range(n):
                for t2 in range(t1+1, n):
                    idx1 = i * n + t1
                    idx2 = i * n + t2
                    Q[idx1, idx2] += penalty
        
        return Q
    
    def decode_solution(self, binary_solution: List[int]) -> List[int]:
        """
        Decode binary solution to route.
        
        Args:
            binary_solution (List[int]): Binary solution from QAOA
        
        Returns:
            List[int]: Ordered list of cities representing the route
        """
        n = self.n_cities
        route = [-1] * n
        binary_matrix = np.array(binary_solution).reshape(n, n)
        
        for t in range(n):
            city = np.argmax(binary_matrix[:, t])
            route[t] = city
            
        return route
