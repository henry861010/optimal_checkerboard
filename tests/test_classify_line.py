import os
import sys
import unittest

SRC_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "src")
)
sys.path.insert(0, SRC_ROOT)

from optimal_checkerboard.algorithms.classify_line import _classify_line


class TestClassifyLine(unittest.TestCase):
    def setUp(self):
        """Configure a caller tolerance capped by representation noise."""
        self.eps = 0.01

    def test_classify_valid_lines(self):
        """Verify valid lines are classified by orientation."""
        lines = [
            [[0, 0, 5], [10, 0, 5]],
            [[2, 0, 1], [2, 10, 1]],
            [[0, 0, 3], [10, 0, 3]],
            [[1, 0, 3], [1, 10, 3]],
        ]

        vertical, horizontal = _classify_line(lines, self.eps)

        self.assertEqual(len(vertical), 2)
        self.assertEqual(len(horizontal), 2)
        self.assertIn([[2, 0, 1], [2, 10, 1]], vertical)
        self.assertIn([[0, 0, 5], [10, 0, 5]], horizontal)

    def test_explicit_large_eps_does_not_accept_near_diagonal_line(self):
        """Verify eps cannot change the topology of a non-axis line."""
        with self.assertRaisesRegex(
            ValueError,
            "Must be strictly vertical or horizontal",
        ):
            _classify_line(
                [[[1.005, 0, 3], [1, 10, 3]]],
                self.eps,
            )

    def test_scale_aware_tolerance_stays_small_for_large_coordinates(self):
        """Verify large coordinate magnitude does not hide a real offset."""
        with self.assertRaisesRegex(
            ValueError,
            "Must be strictly vertical or horizontal",
        ):
            _classify_line(
                [[[1e15, 0, 3], [1e15 + 1.0, 10, 3]]],
            )

    def test_one_ulp_diagonal_is_not_projected_onto_an_axis(self):
        """Every distinct input float remains authoritative topology."""
        x_value = 1e16
        next_x = float.fromhex(x_value.hex()) + 2.0

        with self.assertRaisesRegex(
            ValueError,
            "Must be strictly vertical or horizontal",
        ):
            _classify_line(
                [[[x_value, 0.0, 0.0], [next_x, 100.0, 0.0]]],
                eps=1.0,
            )

    def test_classification_output_is_deterministically_sorted(self):
        """Verify classified lines are sorted by z and fixed coordinate."""
        vertical_z2 = [[3, 0, 2], [3, 10, 2]]
        vertical_z1_x2 = [[2, 0, 1], [2, 10, 1]]
        vertical_z1_x1 = [[1, 0, 1], [1, 10, 1]]
        horizontal_z2 = [[0, 5, 2], [10, 5, 2]]
        horizontal_z1_y2 = [[0, 2, 1], [10, 2, 1]]
        horizontal_z1_y1 = [[0, 1, 1], [10, 1, 1]]

        vertical, horizontal = _classify_line(
            [
                vertical_z2,
                horizontal_z2,
                vertical_z1_x2,
                horizontal_z1_y2,
                vertical_z1_x1,
                horizontal_z1_y1,
            ],
            self.eps,
        )

        self.assertEqual(vertical, [vertical_z1_x1, vertical_z1_x2, vertical_z2])
        self.assertEqual(
            horizontal,
            [horizontal_z1_y1, horizontal_z1_y2, horizontal_z2],
        )

    def test_classify_canonical_z_range_lines(self):
        """Verify canonical z-range lines classify by xy orientation."""
        lines = [
            [[0, 3], [10, 3], [0, 100]],
            [[2, 0], [2, 10], [10, 90]],
        ]

        vertical, horizontal = _classify_line(lines, self.eps)

        self.assertEqual(vertical, [[[2, 0], [2, 10], [10, 90]]])
        self.assertEqual(horizontal, [[[0, 3], [10, 3], [0, 100]]])

    def test_z_coordinate_mismatch(self):
        """Verify that lines with mismatched z coordinates are rejected."""
        with self.assertRaisesRegex(ValueError, "Z coordinates mismatch"):
            _classify_line([[[0, 0, 1], [10, 0, 2]]], self.eps)

    def test_diagonal_line_error(self):
        """Verify that diagonal lines are rejected."""
        with self.assertRaisesRegex(
            ValueError,
            "Must be strictly vertical or horizontal",
        ):
            _classify_line([[[0, 0, 1], [10, 10, 1]]], self.eps)

    def test_single_point_error(self):
        """Verify that zero-length lines are rejected."""
        with self.assertRaisesRegex(
            ValueError,
            "Must be strictly vertical or horizontal",
        ):
            _classify_line([[[5, 5, 1], [5, 5, 1]]], self.eps)


if __name__ == "__main__":
    unittest.main()
