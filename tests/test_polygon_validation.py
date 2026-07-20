import os
import sys
import unittest

SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, SRC_ROOT)

from optimal_checkerboard.algorithms.polygon import normalize_polygon_loops


class TestPolygonValidation(unittest.TestCase):
    def test_valid_hole_is_associated_with_its_unique_hull(self):
        loops = normalize_polygon_loops(
            [
                [[0, 0], [0, 10], [10, 10], [10, 0]],
                [[2, 2], [8, 2], [8, 8], [2, 8]],
            ]
        )

        self.assertEqual(loops[0]["role"], "hull")
        self.assertEqual(loops[1]["role"], "hole")
        self.assertEqual(loops[1]["hull_index"], 0)

    def test_rejects_non_orthogonal_edge(self):
        with self.assertRaisesRegex(ValueError, "non-orthogonal"):
            normalize_polygon_loops(
                [[[0, 0], [0, 2], [2, 1], [2, 0]]]
            )

    def test_rejects_orthogonal_self_intersection(self):
        crossing_loop = [
            [0, 0],
            [0, 3],
            [3, 3],
            [3, 1],
            [-1, 1],
            [-1, 2],
            [2, 2],
            [2, 0],
        ]

        with self.assertRaisesRegex(ValueError, "self-intersecting"):
            normalize_polygon_loops([crossing_loop])

    def test_rejects_hole_outside_hull(self):
        with self.assertRaisesRegex(ValueError, "strictly contained"):
            normalize_polygon_loops(
                [
                    [[0, 0], [0, 10], [10, 10], [10, 0]],
                    [[20, 20], [21, 20], [21, 21], [20, 21]],
                ]
            )

    def test_rejects_hole_touching_hull_boundary(self):
        with self.assertRaisesRegex(ValueError, "intersect or touch"):
            normalize_polygon_loops(
                [
                    [[0, 0], [0, 10], [10, 10], [10, 0]],
                    [[0, 2], [2, 2], [2, 4], [0, 4]],
                ]
            )

    def test_rejects_nested_clockwise_hulls(self):
        with self.assertRaisesRegex(ValueError, "hulls must be disjoint"):
            normalize_polygon_loops(
                [
                    [[0, 0], [0, 10], [10, 10], [10, 0]],
                    [[2, 2], [2, 4], [4, 4], [4, 2]],
                ]
            )


if __name__ == "__main__":
    unittest.main()
