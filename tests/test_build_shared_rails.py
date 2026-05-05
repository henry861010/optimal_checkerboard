import os
import sys
import unittest

SRC_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "src")
)
sys.path.insert(0, SRC_ROOT)

from optimal_checkerboard.algorithms.rail_builder import build_shared_rails


class TestBuildSharedRails(unittest.TestCase):
    def test_same_z_non_overlapping_lines_share_one_rail(self):
        """Verify same-z non-overlapping lines share one checkerboard rail."""
        lines = [
            [[1, 0, 0], [1, 10, 0]],
            [[1.5, 11, 0], [1.5, 20, 0]],
        ]

        result = build_shared_rails(lines, merge_tol=1.0)

        self.assertEqual(result["x_list"], [1.25])
        self.assertEqual(len(result["snap_rules_by_z"][0.0]), 2)

    def test_same_z_overlapping_lines_do_not_share_rail(self):
        """Verify same-z overlapping lines are assigned separate rails."""
        lines = [
            [[1, 0, 0], [1, 10, 0]],
            [[1.5, 5, 0], [1.5, 15, 0]],
        ]

        result = build_shared_rails(lines, merge_tol=1.0)

        self.assertEqual(result["x_list"], [1.0, 1.5])

    def test_same_z_touching_lines_do_not_share_rail(self):
        """Verify touching same-z spans cannot share different targets."""
        lines = [
            [[1, 0, 0], [1, 10, 0]],
            [[1.5, 10, 0], [1.5, 20, 0]],
        ]

        result = build_shared_rails(lines, merge_tol=1.0)

        self.assertEqual(result["x_list"], [1.0, 1.5])

    def test_duplicate_lines_emit_one_snap_rule(self):
        """Verify identical duplicate features do not duplicate snap rules."""
        line = [[1, 0, 0], [1, 10, 0]]

        result = build_shared_rails([line, line], merge_tol=1.0)

        self.assertEqual(result["x_list"], [1.0])
        self.assertEqual(len(result["snap_rules_by_z"][0.0]), 1)

    def test_same_z_intermediate_overlap_blocks_rail_sharing(self):
        """Verify a conflicting intermediate line blocks cross-rail merging."""
        lines = [
            [[10, 0, 0], [10, 10, 0]],
            [[11, 5, 0], [11, 30, 0]],
            [[12, 20, 0], [12, 40, 0]],
        ]

        result = build_shared_rails(lines, merge_tol=2.2)

        self.assertEqual(result["x_list"], [10.0, 11.0, 12.0])

    def test_cross_z_overlapping_lines_can_share_rail(self):
        """Verify cross-z overlapping lines can share one rail."""
        lines = [
            [[0, 1, 0], [0, 10, 0]],
            [[1.5, 5, 10], [1.5, 15, 10]],
        ]

        result = build_shared_rails(lines, merge_tol=2.0)

        self.assertEqual(result["x_list"], [0.75])
        self.assertIn(0.0, result["snap_rules_by_z"])
        self.assertIn(10.0, result["snap_rules_by_z"])


if __name__ == "__main__":
    unittest.main()
