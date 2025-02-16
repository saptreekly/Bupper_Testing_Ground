import numpy as np
from typing import List, Tuple
import random

class Utils:
    @staticmethod
    def generate_random_cities(n_cities: int, 
                             x_range: Tuple[float, float] = (0, 1),
                             y_range: Tuple[float, float] = (0, 1)) -> List[Tuple[float, float]]:
        """
        Generate random city coordinates.
        
        Args:
            n_cities (int): Number of cities
            x_range (Tuple[float, float]): Range for x coordinates
            y_range (Tuple[float, float]): Range for y coordinates
            
        Returns:
            List[Tuple[float, float]]: List of city coordinates
        """
        coordinates = []
        for _ in range(n_cities):
            x = random.uniform(*x_range)
            y = random.uniform(*y_range)
            coordinates.append((x, y))
        return coordinates
    
    @staticmethod
    def calculate_route_length(route: List[int], 
                             distance_matrix: np.ndarray) -> float:
        """
        Calculate total route length.
        
        Args:
            route (List[int]): Ordered list of cities
            distance_matrix (np.ndarray): Matrix of distances between cities
            
        Returns:
            float: Total route length
        """
        total_distance = 0
        for i in range(len(route)):
            city1 = route[i]
            city2 = route[(i + 1) % len(route)]
            total_distance += distance_matrix[city1, city2]
        return total_distance
    
    @staticmethod
    def verify_solution(route: List[int], n_cities: int) -> bool:
        """
        Verify if the solution is valid.
        
        Args:
            route (List[int]): Proposed solution route
            n_cities (int): Number of cities
            
        Returns:
            bool: True if solution is valid
        """
        if len(route) != n_cities:
            return False
        if len(set(route)) != n_cities:
            return False
        if any(city >= n_cities or city < 0 for city in route):
            return False
        return True
