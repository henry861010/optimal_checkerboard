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
        faces = [
            {
                "type": "LINE",
                "dim": [1, 0, 1, 10],
                "bottom_z": 0,
                "top_z": 0,
            }
        ]

        mesher = OptimalMesh25D()
        mesher._set_pattern(faces, element_size=10, ratio=0.1)

        with self.assertRaisesRegex(RuntimeError, "mesh_checkerboard_box"):
            mesher.apply_snap_rules_at_z(0)

    def test_apply_snap_rules_at_z_moves_rail_nodes_to_pattern_lines(self):
        """Verify snap rules move rail nodes back to true pattern positions."""
        faces = [
            {
                "type": "LINE",
                "dim": [1, 0, 1, 10],
                "bottom_z": 0,
                "top_z": 0,
            },
            {
                "type": "LINE",
                "dim": [1.5, 11, 1.5, 20],
                "bottom_z": 0,
                "top_z": 0,
            },
            {
                "type": "LINE",
                "dim": [0, 0, 3, 0],
                "bottom_z": 0,
                "top_z": 0,
            },
            {
                "type": "LINE",
                "dim": [0, 20, 3, 20],
                "bottom_z": 0,
                "top_z": 0,
            },
        ]

        mesher = OptimalMesh25D()
        mesher._set_pattern(faces, element_size=10, ratio=0.1)
        mesh = mesher.mesh_checkerboard_box([0, 0, 3, 20])

        touched = mesher.apply_snap_rules_at_z(0)
        x_values = set(round(float(x), 3) for x in mesh.nodes[:, 0])

        self.assertGreater(touched, 0)
        self.assertIn(1.0, x_values)
        self.assertIn(1.5, x_values)

    def test_apply_snap_rules_mutates_only_matching_node_span(self):
        """Verify span queries leave rail nodes outside the rule intact."""
        faces = [
            {
                "type": "LINE",
                "dim": [1, 0, 1, 5],
                "bottom_z": 0,
                "top_z": 0,
            },
            {
                "type": "LINE",
                "dim": [1.5, 20, 1.5, 25],
                "bottom_z": 0,
                "top_z": 0,
            },
        ]

        mesher = OptimalMesh25D()
        mesher._set_pattern(faces, element_size=5, ratio=0.2)
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

    def test_top_z_restore_is_delayed_until_next_higher_z(self):
        """Verify top-z rules restore the shared rail on the next layer."""
        faces = [
            {
                "type": "LINE",
                "dim": [1, 0, 1, 5],
                "bottom_z": 0,
                "top_z": 10,
            },
            {
                "type": "LINE",
                "dim": [1.5, 20, 1.5, 25],
                "bottom_z": 0,
                "top_z": 10,
            },
        ]

        mesher = OptimalMesh25D()
        mesher._set_pattern(faces, element_size=5, ratio=0.2)
        mesh = mesher.mesh_checkerboard_box([0, 0, 3, 25])
        before = mesh.nodes.copy()

        first_span = (
            np.isclose(before[:, 0], 1.25)
            & (before[:, 1] >= 0)
            & (before[:, 1] <= 5)
        )
        second_span = (
            np.isclose(before[:, 0], 1.25)
            & (before[:, 1] >= 20)
            & (before[:, 1] <= 25)
        )

        mesher.apply_snap_rules_at_z(0)
        self.assertTrue(np.allclose(mesh.nodes[first_span, 0], 1.0))
        self.assertTrue(np.allclose(mesh.nodes[second_span, 0], 1.5))

        self.assertEqual(mesher.apply_snap_rules_at_z(10), 0)
        self.assertTrue(np.allclose(mesh.nodes[first_span, 0], 1.0))
        self.assertTrue(np.allclose(mesh.nodes[second_span, 0], 1.5))

        self.assertEqual(mesher.apply_snap_rules_at_z(10), 0)
        self.assertTrue(np.allclose(mesh.nodes[first_span, 0], 1.0))
        self.assertTrue(np.allclose(mesh.nodes[second_span, 0], 1.5))

        touched = mesher.apply_snap_rules_at_z(11)
        self.assertEqual(touched, int(first_span.sum() + second_span.sum()))
        self.assertTrue(np.allclose(mesh.nodes[first_span, 0], 1.25))
        self.assertTrue(np.allclose(mesh.nodes[second_span, 0], 1.25))

    def test_snap_wins_over_staged_restore_on_same_nodes(self):
        """Verify current bottom snap overwrites a prior top-z restore."""
        faces = [
            {
                "type": "LINE",
                "dim": [1, 0, 1, 5],
                "bottom_z": 0,
                "top_z": 10,
            },
            {
                "type": "LINE",
                "dim": [1.5, 0, 1.5, 5],
                "bottom_z": 11,
                "top_z": 20,
            },
        ]

        mesher = OptimalMesh25D()
        mesher._set_pattern(faces, element_size=5, ratio=0.2)
        mesh = mesher.mesh_checkerboard_box([0, 0, 3, 5])
        before = mesh.nodes.copy()
        shared_span = (
            np.isclose(before[:, 0], 1.25)
            & (before[:, 1] >= 0)
            & (before[:, 1] <= 5)
        )

        mesher.apply_snap_rules_at_z(0)
        mesher.apply_snap_rules_at_z(10)
        mesher.apply_snap_rules_at_z(11)

        self.assertTrue(np.allclose(mesh.nodes[shared_span, 0], 1.5))

    def test_cross_z_box_corner_snaps_to_next_face(self):
        """Verify shared-rail corners snap when both axes move at one z."""
        faces = [
            {
                "type": "BOX",
                "dim": [0, 0, 10, 10],
                "bottom_z": 0,
                "top_z": 10,
            },
            {
                "type": "BOX",
                "dim": [2, 1, 8, 5],
                "bottom_z": 11,
                "top_z": 20,
            },
        ]

        mesher = OptimalMesh25D()
        mesher._set_pattern(faces, element_size=11.0, ratio=0.2)
        mesh = mesher.mesh_checkerboard_box([-1, -1, 11, 11])

        mesher.apply_snap_rules_at_z(0)
        mesher.apply_snap_rules_at_z(11)

        points = {
            (round(float(x), 6), round(float(y), 6))
            for x, y in mesh.nodes[:, :2]
        }

        self.assertIn((2.0, 1.0), points)
        self.assertIn((8.0, 1.0), points)
        self.assertIn((2.0, 5.0), points)
        self.assertIn((8.0, 5.0), points)
        self.assertNotIn((0.0, 0.0), points)
        self.assertNotIn((10.0, 0.0), points)

    def test_apply_snap_rules_rejects_bad_nodes_shape(self):
        """Verify callers receive a clear error for invalid node arrays."""
        faces = [
            {
                "type": "LINE",
                "dim": [1, 0, 1, 10],
                "bottom_z": 0,
                "top_z": 0,
            }
        ]

        mesher = OptimalMesh25D()
        mesher._set_pattern(faces, element_size=10, ratio=0.1)
        mesher.mesh_checkerboard_box([0, 0, 3, 10])

        with self.assertRaisesRegex(ValueError, "shape"):
            mesher.apply_snap_rules_at_z(0, nodes=np.zeros(1))


if __name__ == "__main__":
    unittest.main()
