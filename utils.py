import numpy as np
from typing import List, Tuple
import random
import logging

logger = logging.getLogger(__name__)

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
        Calculate total route length (Hamiltonian path, not cyclic).

        Args:
            route (List[int]): Ordered list of cities
            distance_matrix (np.ndarray): Matrix of distances between cities

        Returns:
            float: Total route length (without return to start)
        """
        logger.debug("Calculating route length for route: %s", route)
        logger.debug("Distance matrix:\n%s", distance_matrix)

        total_distance = 0
        # Only iterate to n-1 to avoid counting return to start
        for i in range(len(route) - 1):
            city1 = route[i]
            city2 = route[i + 1]
            distance = distance_matrix[city1, city2]
            logger.debug("Distance from city %d to city %d: %.3f", city1, city2, distance)
            total_distance += distance

        logger.debug("Total route length (Hamiltonian path): %.3f", total_distance)
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
            logger.warning("Invalid route length: expected %d cities, got %d", n_cities, len(route))
            return False
        if len(set(route)) != n_cities:
            logger.warning("Duplicate cities in route: %s", route)
            return False
        if any(city >= n_cities or city < 0 for city in route):
            logger.warning("Invalid city indices in route: %s", route)
            return False
        logger.debug("Valid route found: %s", route)
        return True