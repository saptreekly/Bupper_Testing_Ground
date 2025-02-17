import unittest
from example import manhattan_distance_with_obstacles
import logging

logging.basicConfig(level=logging.DEBUG,
                   format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

class TestPathfinding(unittest.TestCase):
    def setUp(self):
        """Set up test cases with different grid configurations."""
        self.grid_size = 16

    def test_direct_path_no_obstacles(self):
        """Test path finding with no obstacles - should find direct path."""
        obstacles = set()
        start = (0, 0)
        end = (2, 0)

        cost, path = manhattan_distance_with_obstacles(start, end, obstacles, self.grid_size)

        self.assertEqual(cost, 2)  # Direct horizontal distance
        self.assertEqual(path[0], start)  # Should start at start position
        self.assertEqual(path[-1], end)   # Should end at end position
        logger.info(f"Direct path test passed: {path}")

    def test_diagonal_path(self):
        """Test path finding for diagonal movement."""
        obstacles = set()
        start = (0, 0)
        end = (2, 2)

        cost, path = manhattan_distance_with_obstacles(start, end, obstacles, self.grid_size)

        self.assertEqual(cost, 4)  # Manhattan distance for diagonal
        self.assertEqual(path[0], start)
        self.assertEqual(path[-1], end)
        logger.info(f"Diagonal path test passed: {path}")


if __name__ == '__main__':
    unittest.main(verbosity=2)