import os
import sys
import unittest

SRC_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "src")
)
sys.path.insert(0, SRC_ROOT)

from optimal_checkerboard.algorithms.classify_line import _classify_line
from optimal_checkerboard.algorithms.is_related import _is_related
from optimal_checkerboard.algorithms.merge_lines_xy import _merge_lines_xy
from optimal_checkerboard.algorithms.merge_lines_z import _merge_lines_z
from optimal_checkerboard.algorithms.unique_lines import _unique_lines


class TestClassifyLine(unittest.TestCase):
    def setUp(self):
        """Configure the default classification tolerance."""
        self.eps = 0.01

    def test_classify_valid_lines(self):
        """Verify valid lines are classified by orientation."""
        lines = [
            [[0, 0, 5], [10, 0, 5]],
            [[2, 0, 1], [2, 10, 1]],
            [[0, 0.005, 3], [10, 0, 3]],
            [[1.005, 0, 3], [1, 10, 3]],
        ]

        vertical, horizontal = _classify_line(lines, self.eps)

        self.assertEqual(len(vertical), 2)
        self.assertEqual(len(horizontal), 2)
        self.assertIn([[2, 0, 1], [2, 10, 1]], vertical)
        self.assertIn([[0, 0, 5], [10, 0, 5]], horizontal)

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


class TestRelatedLines(unittest.TestCase):
    def test_vertical_related_and_unrelated_cases(self):
        """Verify vertical related-line checks for nearby separated spans."""
        line = [[0, 0, 0], [0, 10, 0]]

        self.assertTrue(
            _is_related(line, [[0.5, 12, 0], [0.5, 15, 0]], l=1.0, v_h=0)
        )
        self.assertFalse(
            _is_related(line, [[0.5, 10, 0], [0.5, 15, 0]], l=1.0, v_h=0)
        )
        self.assertFalse(
            _is_related(line, [[0.5, 10.5, 0], [0.5, 15, 0]], l=1.0, v_h=0)
        )
        self.assertFalse(
            _is_related(line, [[0.5, 5, 0], [0.5, 15, 0]], l=1.0, v_h=0)
        )
        self.assertFalse(
            _is_related(line, [[5, 12, 0], [5, 15, 0]], l=1.0, v_h=0)
        )

    def test_horizontal_related_and_unrelated_cases(self):
        """Verify horizontal related-line checks for nearby separated spans."""
        line = [[0, 5, 0], [10, 5, 0]]

        self.assertTrue(
            _is_related(
                line,
                [[-10, 5.5, 0], [-2, 5.5, 0]],
                l=1.0,
                v_h=1,
            )
        )
        self.assertFalse(
            _is_related(
                line,
                [[10, 5.5, 0], [15, 5.5, 0]],
                l=1.0,
                v_h=1,
            )
        )
        self.assertFalse(
            _is_related(line, [[10.5, 5.5, 0], [15, 5.5, 0]], l=1.0, v_h=1)
        )
        self.assertFalse(
            _is_related(line, [[8, 5.5, 0], [15, 5.5, 0]], l=1.0, v_h=1)
        )

    def test_invalid_relation_inputs_return_false(self):
        """Verify invalid or overlapping relation inputs return False."""
        self.assertFalse(
            _is_related(
                [[0, 0, 0], [0, 10, 0]],
                [[1, 11, 0], [5, 11, 0]],
                l=1.0,
                v_h=0,
            )
        )
        self.assertFalse(
            _is_related(
                [[0, 0, 0], [0, 10, 0]],
                [[0, 5, 0], [0, 15, 0]],
                l=1.0,
                v_h=0,
            )
        )


class TestMergeLinesByAxis(unittest.TestCase):
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


class TestUniqueLines2D(unittest.TestCase):
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
