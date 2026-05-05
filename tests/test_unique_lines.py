import os
import sys
import unittest

SRC_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "src")
)
sys.path.insert(0, SRC_ROOT)

from optimal_checkerboard.algorithms.unique_lines import _unique_lines


class TestUniqueLines(unittest.TestCase):
    def test_no_duplicates(self):
        """Verify distinct lines are preserved."""
        group_lines = {
            1: [
                [[0.0, 0.0, 1], [5.0, 0.0, 1]],
                [[0.0, 1.0, 1], [5.0, 1.0, 1]],
            ]
        }

        result = _unique_lines(group_lines)

        self.assertEqual(len(result[1]), 2)

    def test_same_layer_duplicate(self):
        """Verify duplicate lines in the same layer are removed."""
        line = [[0.0, 0.0, 1], [5.0, 0.0, 1]]

        result = _unique_lines({1: [line, line]})

        self.assertEqual(result[1], [line])

    def test_cross_layer_duplicate(self):
        """Verify duplicate lines across layers keep the first occurrence."""
        l1 = [[0.0, 0.0, 1], [5.0, 0.0, 1]]
        l2 = [[0.0, 0.0, 5], [5.0, 0.0, 5]]

        result = _unique_lines({1: [l1], 5: [l2]})

        self.assertEqual(len(result), 1)
        self.assertIn(1, result)
        self.assertNotIn(5, result)

    def test_reversed_direction(self):
        """Verify reversed line direction is treated as the same line."""
        l1 = [[0.0, 0.0, 1], [5.0, 5.0, 1]]
        l2 = [[5.0, 5.0, 1], [0.0, 0.0, 1]]

        result = _unique_lines({1: [l1, l2]})

        self.assertEqual(len(result[1]), 1)

    def test_floating_point_noise(self):
        """Verify tiny coordinate noise does not create duplicate lines."""
        l1 = [[0.0, 0.0, 1], [5.0, 5.0, 1]]
        l2 = [[0.0, 0.0, 1], [5.000000001, 5.0, 1]]

        result = _unique_lines({1: [l1, l2]}, decimals=4)

        self.assertEqual(len(result[1]), 1)

    def test_different_cross_layer_lines_are_kept(self):
        """Verify different lines across layers are preserved."""
        l1 = [[0.0, 0.0, 1], [5.0, 5.0, 1]]
        l2 = [[1.0, 2.0, 1], [6.0, 8.0, 1]]

        result = _unique_lines({1: [l1], 5: [l2]}, decimals=4)

        self.assertEqual(len(result), 2)
        self.assertIn(1, result)
        self.assertIn(5, result)


if __name__ == "__main__":
    unittest.main()
