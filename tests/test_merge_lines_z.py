import os
import sys
import unittest

SRC_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "src")
)
sys.path.insert(0, SRC_ROOT)

from optimal_checkerboard.algorithms.merge_lines_z import _merge_lines_z


class TestMergeLinesZ(unittest.TestCase):
    def test_single_z_multiple_groups(self):
        """Verify same-z line sets are not merged into one z group."""
        groups_z = {
            1: [
                [[[0, 0], [0, 10]], [[1, 11], [1, 20]]],
                [[[2, 0], [2, 10]], [[3, 11], [3, 20]]],
            ]
        }

        result, _ = _merge_lines_z(groups_z, element_size=4, v_h=0)

        self.assertEqual(len(result), 2)
        self.assertIn(1, result[0])
        self.assertIn(1, result[1])

    def test_cross_z_merge(self):
        """Verify close line sets from different z levels can merge."""
        groups_z = {
            1: [[[[0, 0], [0, 10]], [[1, 11], [1, 20]]]],
            2: [[[[1, 0], [1, 10]], [[2, 11], [2, 20]]]],
        }

        result, _ = _merge_lines_z(groups_z, element_size=4, v_h=0)

        self.assertEqual(len(result), 1)
        self.assertIn(1, result[0])
        self.assertIn(2, result[0])

    def test_cross_z_merge_updates_bounds(self):
        """Verify returned bounds expand to include merged z groups."""
        groups_z = {
            1: [[[[0, 0], [0, 10]], [[2, 11], [2, 20]]]],
            2: [[[[1, 0], [1, 10]], [[3, 11], [3, 20]]]],
        }

        result, bounds = _merge_lines_z(groups_z, element_size=5, v_h=0)

        self.assertEqual(len(result), 1)
        self.assertEqual(bounds, [[0.0, 3.0]])

    def test_cross_z_no_merge(self):
        """Verify distant line sets from different z levels stay separate."""
        groups_z = {
            1: [[[[0, 0], [0, 10]], [[1, 11], [1, 20]]]],
            2: [[[[30, 0], [30, 10]], [[31, 11], [31, 20]]]],
        }

        result, _ = _merge_lines_z(groups_z, element_size=4, v_h=0)

        self.assertEqual(len(result), 2)

    def test_complex_multi_level(self):
        """Verify multi-level z grouping creates the expected group count."""
        groups_z = {
            1: [
                [[[0, 0], [0, 10]], [[1, 11], [1, 20]]],
                [[[20, 0], [20, 10]], [[21, 11], [21, 20]]],
            ],
            2: [
                [[[0, 0], [0, 10]], [[1, 11], [1, 20]]],
                [[[30, 0], [30, 10]], [[31, 11], [31, 20]]],
            ],
        }

        result, _ = _merge_lines_z(groups_z, element_size=4, v_h=0)

        self.assertEqual(len(result), 3)


if __name__ == "__main__":
    unittest.main()
