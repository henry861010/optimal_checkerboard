import os
import sys
import unittest

SRC_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "src")
)
sys.path.insert(0, SRC_ROOT)

import numpy as np

from optimal_checkerboard import OptimalMesh25D


class TestApplySnapRulesAtZ(unittest.TestCase):
    def test_apply_snap_rules_requires_mesh(self):
        """Verify snap rules cannot be applied before a mesh is indexed."""
        faces = [{"type": "LINE", "dim": [[1, 0, 0], [1, 10, 0]]}]

        mesher = OptimalMesh25D()
        mesher.set_pattern(faces, element_size=10, ratio=0.1)

        with self.assertRaisesRegex(RuntimeError, "mesh_checkerboard_box"):
            mesher.apply_snap_rules_at_z(0)

    def test_apply_snap_rules_at_z_moves_rail_nodes_to_pattern_lines(self):
        """Verify snap rules move rail nodes back to true pattern positions."""
        faces = [
            {
                "type": "POLYGON",
                "dim": [
                    [[1, 0, 0], [1, 10, 0]],
                    [[1.5, 11, 0], [1.5, 20, 0]],
                    [[0, 0, 0], [3, 0, 0]],
                    [[0, 20, 0], [3, 20, 0]],
                ],
            }
        ]

        mesher = OptimalMesh25D()
        mesher.set_pattern(faces, element_size=10, ratio=0.1)
        mesh = mesher.mesh_checkerboard_box([0, 0, 3, 20])

        touched = mesher.apply_snap_rules_at_z(0)
        x_values = set(round(float(x), 3) for x in mesh.nodes[:, 0])

        self.assertGreater(touched, 0)
        self.assertIn(1.0, x_values)
        self.assertIn(1.5, x_values)

    def test_apply_snap_rules_mutates_only_matching_node_span(self):
        """Verify span queries leave rail nodes outside the rule intact."""
        faces = [
            {"type": "LINE", "dim": [[1, 0, 0], [1, 5, 0]]},
            {"type": "LINE", "dim": [[1.5, 20, 0], [1.5, 25, 0]]},
        ]

        mesher = OptimalMesh25D()
        mesher.set_pattern(faces, element_size=5, ratio=0.2)
        mesh = mesher.mesh_checkerboard_box([0, 0, 3, 25])
        before = mesh.nodes.copy()

        touched = mesher.apply_snap_rules_at_z(0)

        before_x = before[:, 0]
        before_y = before[:, 1]
        after_x = mesh.nodes[:, 0]

        first_span = (
            np.isclose(before_x, 1.25)
            & (before_y >= 0)
            & (before_y <= 5)
        )
        untouched_span = (
            np.isclose(before_x, 1.25)
            & (before_y > 5)
            & (before_y < 20)
        )
        second_span = (
            np.isclose(before_x, 1.25)
            & (before_y >= 20)
            & (before_y <= 25)
        )

        self.assertEqual(touched, int(first_span.sum() + second_span.sum()))
        self.assertTrue(np.allclose(after_x[first_span], 1.0))
        self.assertTrue(np.allclose(after_x[untouched_span], 1.25))
        self.assertTrue(np.allclose(after_x[second_span], 1.5))

    def test_apply_snap_rules_rejects_bad_nodes_shape(self):
        """Verify callers receive a clear error for invalid node arrays."""
        faces = [{"type": "LINE", "dim": [[1, 0, 0], [1, 10, 0]]}]

        mesher = OptimalMesh25D()
        mesher.set_pattern(faces, element_size=10, ratio=0.1)
        mesher.mesh_checkerboard_box([0, 0, 3, 10])

        with self.assertRaisesRegex(ValueError, "shape"):
            mesher.apply_snap_rules_at_z(0, nodes=np.zeros(1))


if __name__ == "__main__":
    unittest.main()
