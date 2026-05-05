import os
import sys
import unittest

SRC_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "src")
)
sys.path.insert(0, SRC_ROOT)

from optimal_checkerboard.mesh.checkerboard_mesh_box import (
    checkerboard_mesh_box,
)


class TestCheckerboardMeshBox(unittest.TestCase):
    def test_required_axis_coordinates_are_preserved_when_subdividing(self):
        """Verify mandatory rails remain present after interval subdivision."""
        nodes, elements = checkerboard_mesh_box(
            [0, 4, 14],
            [0, 6],
            element_size=5,
        )

        x_values = sorted({round(float(value), 6) for value in nodes[:, 0]})
        y_values = sorted({round(float(value), 6) for value in nodes[:, 1]})

        self.assertEqual(x_values, [0.0, 4.0, 9.0, 14.0])
        self.assertEqual(y_values, [0.0, 3.0, 6.0])
        self.assertEqual(nodes.shape, (12, 3))
        self.assertEqual(elements.shape, (6, 4))

    def test_short_intervals_still_create_one_element(self):
        """Verify intervals smaller than element_size still produce a cell."""
        nodes, elements = checkerboard_mesh_box(
            [0, 1],
            [0, 1],
            element_size=10,
        )

        self.assertEqual(nodes.shape, (4, 3))
        self.assertEqual(elements.tolist(), [[0, 1, 3, 2]])


if __name__ == "__main__":
    unittest.main()
