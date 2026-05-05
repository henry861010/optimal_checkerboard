import os
import sys
import unittest

SRC_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "src")
)
sys.path.insert(0, SRC_ROOT)

from optimal_checkerboard.algorithms.merge_lines_xy import _merge_lines_xy


class TestMergeLinesXY(unittest.TestCase):
    def setUp(self):
        """Configure the default merge tolerance for xy grouping tests."""
        self.eps = 0.01
        self.element_size = 1.0

    def test_empty_input(self):
        """Verify that empty line input returns no groups."""
        self.assertEqual(
            _merge_lines_xy([], self.element_size, v_h=0, eps=self.eps),
            [],
        )

    def test_single_line_input(self):
        """Verify that one line becomes one isolated group."""
        lines = [[[0.0, 0.0, 0.0], [0.0, 5.0, 0.0]]]
        self.assertEqual(
            _merge_lines_xy(lines, self.element_size, v_h=0, eps=self.eps),
            [lines],
        )

    def test_two_line_input_keeps_both_boundary_lines_separate(self):
        """Verify two-line inputs do not merge boundary rails."""
        v1 = [[0.0, 0.0, 0.0], [0.0, 5.0, 0.0]]
        v2 = [[0.5, 6.0, 0.0], [0.5, 11.0, 0.0]]

        groups = _merge_lines_xy(
            [v2, v1],
            self.element_size,
            v_h=0,
            eps=self.eps,
        )

        self.assertEqual(groups, [[v1], [v2]])

    def test_vertical_chaining_and_isolation(self):
        """Verify vertical line grouping with boundary-line isolation."""
        v1 = [[0.0, 0.0, 0.0], [0.0, 5.0, 0.0]]
        v2 = [[1.0, 6.0, 0.0], [1.0, 11.0, 0.0]]
        v3 = [[2.0, 12.0, 0.0], [2.0, 17.0, 0.0]]
        v4 = [[3.0, 18.0, 0.0], [3.0, 23.0, 0.0]]
        v5 = [[10.0, 50.0, 0.0], [10.0, 55.0, 0.0]]

        groups = _merge_lines_xy(
            [v1, v2, v3, v4, v5],
            self.element_size,
            v_h=0,
            eps=self.eps,
        )

        self.assertEqual(groups, [[v1], [v2, v3, v4], [v5]])

    def test_horizontal_chaining_and_isolation(self):
        """Verify horizontal line grouping with boundary-line isolation."""
        h1 = [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]
        h2 = [[6.0, 1.0, 0.0], [11.0, 1.0, 0.0]]
        h3 = [[12.0, 2.0, 0.0], [17.0, 2.0, 0.0]]
        h4 = [[50.0, 10.0, 0.0], [55.0, 10.0, 0.0]]

        groups = _merge_lines_xy(
            [h1, h2, h3, h4],
            self.element_size,
            v_h=1,
            eps=self.eps,
        )

        self.assertEqual(groups, [[h1], [h2, h3], [h4]])

    def test_unrelated_middle_lines(self):
        """Verify unrelated middle lines remain separate groups."""
        v1 = [[0.0, 0.0, 0.0], [0.0, 5.0, 0.0]]
        v2 = [[5.0, 10.0, 0.0], [5.0, 12.0, 0.0]]
        v3 = [[10.0, 80.0, 0.0], [10.0, 585.0, 0.0]]

        groups = _merge_lines_xy([v1, v2, v3], 20, v_h=0, eps=self.eps)

        self.assertEqual(groups, [[v1], [v2], [v3]])


if __name__ == "__main__":
    unittest.main()
