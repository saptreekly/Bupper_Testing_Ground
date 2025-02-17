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
        self.assertEqual(path, [(0, 0), (1, 0), (2, 0)])  # Direct path
        logger.info(f"Direct path test passed: {path}")

    def test_simple_vertical_obstacle(self):
        """Test path finding with a simple vertical obstacle requiring a detour."""
        # Create vertical obstacle that blocks direct path
        obstacles = {('v', 1, 0)}  # Vertical obstacle at x=1
        start = (0, 0)
        end = (2, 0)
        
        cost, path = manhattan_distance_with_obstacles(start, end, obstacles, self.grid_size)
        
        # Should go around the obstacle (up and then down)
        expected_path = [(0, 0), (0, 1), (1, 1), (2, 1), (2, 0)]
        self.assertEqual(cost, 4)  # Detour costs more
        self.assertEqual(path, expected_path)
        logger.info(f"Vertical obstacle test passed: {path}")

    def test_complex_obstacle_configuration(self):
        """Test path finding with multiple obstacles forming a maze-like pattern."""
        obstacles = {
            ('v', 1, 0), ('v', 1, 1),  # Vertical wall
            ('h', 0, 2), ('h', 1, 2),  # Horizontal wall
            ('v', 3, 1), ('v', 3, 2)   # Another vertical wall
        }
        start = (0, 0)
        end = (4, 0)
        
        cost, path = manhattan_distance_with_obstacles(start, end, obstacles, self.grid_size)
        
        self.assertIsNotNone(path)
        self.assertGreater(len(path), 4)  # Path should be longer due to obstacles
        
        # Verify path never crosses obstacles
        self._verify_no_obstacle_crossing(path, obstacles)
        logger.info(f"Complex obstacle test passed: {path}")

    def test_no_valid_path(self):
        """Test when no valid path exists between points."""
        # Create a complete wall of obstacles
        obstacles = {('v', 1, i) for i in range(5)}  # Vertical wall
        start = (0, 2)
        end = (2, 2)
        
        cost, path = manhattan_distance_with_obstacles(start, end, obstacles, self.grid_size)
        
        self.assertEqual(cost, self.grid_size * 10)  # Should return high cost
        self.assertEqual(path, [])  # Should return empty path
        logger.info("No valid path test passed")

    def test_diagonal_obstacle_avoidance(self):
        """Test that paths properly avoid diagonal obstacles."""
        obstacles = {
            ('v', 1, 1), ('h', 1, 1),  # Corner obstacle
            ('v', 2, 2), ('h', 2, 2)   # Another corner obstacle
        }
        start = (0, 0)
        end = (3, 3)
        
        cost, path = manhattan_distance_with_obstacles(start, end, obstacles, self.grid_size)
        
        self._verify_no_obstacle_crossing(path, obstacles)
        self._verify_grid_aligned_movement(path)
        logger.info(f"Diagonal avoidance test passed: {path}")

    def _verify_no_obstacle_crossing(self, path, obstacles):
        """Helper method to verify path never crosses obstacles."""
        if not path:
            return
        
        for i in range(len(path) - 1):
            current = path[i]
            next_pos = path[i + 1]
            
            # Check vertical movement
            if current[0] == next_pos[0]:
                x = current[0]
                y_min = min(current[1], next_pos[1])
                y_max = max(current[1], next_pos[1])
                for y in range(y_min, y_max):
                    self.assertNotIn(('v', x, y), obstacles, 
                                   f"Path crosses vertical obstacle at ({x}, {y})")
            
            # Check horizontal movement
            if current[1] == next_pos[1]:
                y = current[1]
                x_min = min(current[0], next_pos[0])
                x_max = max(current[0], next_pos[0])
                for x in range(x_min, x_max):
                    self.assertNotIn(('h', x, y), obstacles,
                                   f"Path crosses horizontal obstacle at ({x}, {y})")

    def _verify_grid_aligned_movement(self, path):
        """Helper method to verify all movements are grid-aligned (no diagonals)."""
        if not path:
            return
        
        for i in range(len(path) - 1):
            current = path[i]
            next_pos = path[i + 1]
            
            # Either x or y should be the same for grid-aligned movement
            self.assertTrue(
                current[0] == next_pos[0] or current[1] == next_pos[1],
                f"Non-grid-aligned movement detected between {current} and {next_pos}"
            )

if __name__ == '__main__':
    unittest.main(verbosity=2)
