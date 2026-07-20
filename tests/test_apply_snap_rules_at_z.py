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
    def test_snap_state_cannot_switch_node_buffers_before_reset(self):
        """Sparse restore ids must never leak between independent arrays."""
        faces = [
            {
                "type": "LINE",
                "dim": [1.0, 0.0, 1.0, 5.0],
                "bottom_z": 0.0,
                "top_z": 1.0,
            },
            {
                "type": "LINE",
                "dim": [1.5, 10.0, 1.5, 15.0],
                "bottom_z": 0.0,
                "top_z": 1.0,
            },
        ]
        mesher = OptimalMesh25D()
        mesher._set_pattern(
            faces,
            element_size=5.0,
            ratio=0.2,
            mesh_domain={"type": "BOX", "dim": [0.0, 0.0, 3.0, 15.0]},
        )
        mesh = mesher.mesh_checkerboard()
        baseline = mesh.nodes.copy()
        external = baseline.copy()

        self.assertGreater(mesher.apply_snap_rules_at_z(0.0), 0)
        with self.assertRaisesRegex(RuntimeError, "different node buffer"):
            mesher.apply_snap_rules_at_z(2.0, nodes=external)

        np.testing.assert_array_equal(external, baseline)
        self.assertGreater(mesher.reset_snap_state(), 0)
        self.assertEqual(mesher.apply_snap_rules_at_z(2.0, nodes=external), 0)
        np.testing.assert_array_equal(mesh.nodes, baseline)

    def test_default_z_matching_never_starts_a_nearby_event_early(self):
        """Physical mutation uses exact z topology unless explicitly opted in."""
        event_z = 1.0000005
        faces = [
            {
                "type": "LINE",
                "dim": [1.0, 0.0, 1.0, 5.0],
                "bottom_z": event_z,
                "top_z": event_z,
            },
            {
                "type": "LINE",
                "dim": [1.5, 10.0, 1.5, 15.0],
                "bottom_z": 2.0,
                "top_z": 2.0,
            },
        ]
        mesher = OptimalMesh25D()
        mesher._set_pattern(
            faces,
            element_size=5.0,
            ratio=0.2,
            mesh_domain={"type": "BOX", "dim": [0.0, 0.0, 3.0, 15.0]},
        )
        mesh = mesher.mesh_checkerboard()
        baseline = mesh.nodes.copy()

        self.assertEqual(mesher.apply_snap_rules_at_z(1.0), 0)
        np.testing.assert_array_equal(mesh.nodes, baseline)
        self.assertGreater(mesher.apply_snap_rules_at_z(event_z), 0)

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

        with self.assertRaisesRegex(RuntimeError, "mesh_checkerboard"):
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
        mesher._set_pattern(
            faces,
            element_size=10,
            ratio=0.1,
            mesh_domain={"type": "BOX", "dim": [0, 0, 3, 20]},
        )
        mesh = mesher.mesh_checkerboard()

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
        mesher._set_pattern(
            faces,
            element_size=5,
            ratio=0.2,
            mesh_domain={"type": "BOX", "dim": [0, 0, 3, 25]},
        )
        mesh = mesher.mesh_checkerboard()
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
        mesher._set_pattern(
            faces,
            element_size=5,
            ratio=0.2,
            mesh_domain={"type": "BOX", "dim": [0, 0, 3, 25]},
        )
        mesh = mesher.mesh_checkerboard()
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
        mesher._set_pattern(
            faces,
            element_size=5,
            ratio=0.2,
            mesh_domain={"type": "BOX", "dim": [0, 0, 3, 5]},
        )
        mesh = mesher.mesh_checkerboard()
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

    def test_overlapping_same_target_feature_does_not_restore_early(self):
        """Verify a shorter duplicate cannot restore a still-active feature."""
        faces = [
            {
                "type": "LINE",
                "dim": [0, 0, 0, 10],
                "bottom_z": 0,
                "top_z": 5,
            },
            {
                "type": "LINE",
                "dim": [0, 0, 0, 10],
                "bottom_z": 0,
                "top_z": 10,
            },
            {
                "type": "LINE",
                "dim": [1, 20, 1, 30],
                "bottom_z": 0,
                "top_z": 10,
            },
        ]

        mesher = OptimalMesh25D()
        mesher._set_pattern(
            faces,
            element_size=10,
            ratio=0.2,
            mesh_domain={"type": "BOX", "dim": [-1, 0, 2, 30]},
        )
        mesh = mesher.mesh_checkerboard()
        shared_span = (
            np.isclose(mesh.nodes[:, 0], 0.5)
            & (mesh.nodes[:, 1] >= 0)
            & (mesh.nodes[:, 1] <= 10)
        )

        mesher.apply_snap_rules_at_z(0)
        mesher.apply_snap_rules_at_z(5)
        mesher.apply_snap_rules_at_z(6)

        self.assertTrue(np.allclose(mesh.nodes[shared_span, 0], 0.0))

        mesher.apply_snap_rules_at_z(10)
        mesher.apply_snap_rules_at_z(11)

        self.assertTrue(np.allclose(mesh.nodes[shared_span, 0], 0.5))

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
        mesher._set_pattern(
            faces,
            element_size=11.0,
            ratio=0.2,
            mesh_domain={"type": "BOX", "dim": [-1, -1, 11, 11]},
        )
        mesh = mesher.mesh_checkerboard()

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
        mesher._set_pattern(
            faces,
            element_size=10,
            ratio=0.1,
            mesh_domain={"type": "BOX", "dim": [0, 0, 3, 10]},
        )
        mesher.mesh_checkerboard()

        with self.assertRaisesRegex(ValueError, "shape"):
            mesher.apply_snap_rules_at_z(0, nodes=np.zeros(1))

    def test_apply_uses_raw_high_precision_z_for_distinct_buckets(self):
        """Verify applying raw z values selects the intended rounded event."""
        faces = [
            {
                "type": "LINE",
                "dim": [1, 0, 1, 5],
                "bottom_z": 1.23444,
                "top_z": 1.23444,
            },
            {
                "type": "LINE",
                "dim": [1.5, 6, 1.5, 11],
                "bottom_z": 1.23446,
                "top_z": 1.23446,
            },
        ]
        mesher = OptimalMesh25D()
        mesher._set_pattern(
            faces,
            element_size=5,
            ratio=0.2,
            mesh_domain={"type": "BOX", "dim": [0, 0, 3, 11]},
        )
        mesh = mesher.mesh_checkerboard()
        baseline = mesh.nodes.copy()

        first_touched, first_node_ids = mesher.apply_snap_rules_at_z(
            1.23444,
            return_touched_node_ids=True,
        )
        restored = mesher.reset_snap_state()
        self.assertGreater(restored, 0)
        np.testing.assert_allclose(
            mesh.nodes[first_node_ids, :2],
            baseline[first_node_ids, :2],
        )
        second_touched, second_node_ids = mesher.apply_snap_rules_at_z(
            1.23446,
            return_touched_node_ids=True,
        )

        self.assertGreater(first_touched, 0)
        self.assertGreater(second_touched, 0)
        self.assertTrue(np.all(baseline[first_node_ids, 1] <= 5))
        self.assertTrue(np.all(baseline[second_node_ids, 1] >= 6))

    def test_active_partial_overlap_wins_over_earlier_restore(self):
        """Verify a continuing feature is re-applied after a partial restore."""
        faces = [
            {
                "type": "LINE",
                "dim": [1, 0, 1, 10],
                "bottom_z": 0,
                "top_z": 20,
            },
            {
                "type": "LINE",
                "dim": [1, 5, 1, 15],
                "bottom_z": 5,
                "top_z": 10,
            },
            {
                "type": "LINE",
                "dim": [1.5, 20, 1.5, 30],
                "bottom_z": 30,
                "top_z": 40,
            },
        ]
        mesher = OptimalMesh25D()
        mesher._set_pattern(
            faces,
            element_size=5,
            ratio=0.2,
            mesh_domain={"type": "BOX", "dim": [0, 0, 3, 30]},
        )
        mesh = mesher.mesh_checkerboard()

        mesher.apply_snap_rules_at_z(0)
        mesher.apply_snap_rules_at_z(5)
        mesher.apply_snap_rules_at_z(10)
        mesher.apply_snap_rules_at_z(15)

        active_span = (
            np.isclose(mesh.nodes[:, 0], 1.0)
            & (mesh.nodes[:, 1] >= 0)
            & (mesh.nodes[:, 1] <= 10)
        )
        restored_tail = (
            np.isclose(mesh.nodes[:, 0], 1.25)
            & np.isclose(mesh.nodes[:, 1], 15)
        )
        self.assertEqual(int(active_span.sum()), 3)
        self.assertEqual(int(restored_tail.sum()), 1)

        mesher.apply_snap_rules_at_z(20)
        self.assertEqual(
            int(
                (
                    np.isclose(mesh.nodes[:, 0], 1.0)
                    & (mesh.nodes[:, 1] <= 10)
                ).sum()
            ),
            3,
        )

    def test_sub_picometer_snap_is_tracked_and_exactly_restored(self):
        """A real displacement below old isclose tolerances is not lost."""
        second_target = 1e-12
        faces = [
            {
                "type": "LINE",
                "dim": [0.0, 0.0, 0.0, 1.0],
                "bottom_z": 0.0,
                "top_z": 0.0,
            },
            {
                "type": "LINE",
                "dim": [second_target, 0.0, second_target, 1.0],
                "bottom_z": 2.0,
                "top_z": 2.0,
            },
        ]
        mesher = OptimalMesh25D()
        mesher._set_pattern(
            faces,
            element_size=1.0,
            ratio=1.0,
            mesh_domain={"type": "BOX", "dim": [-1.0, 0.0, 1.0, 1.0]},
        )
        mesh = mesher.mesh_checkerboard()
        first_rule = mesher.snap_rules_by_z[0.0][0]
        rail_coord = float(
            mesher.rails["x"][first_rule["rail_id"]]["coord"]
        )
        self.assertEqual(rail_coord, 0.5e-12)

        touched, node_ids = mesher.apply_snap_rules_at_z(
            0.0,
            return_touched_node_ids=True,
        )
        self.assertGreater(touched, 0)
        self.assertTrue(np.all(mesh.nodes[node_ids, 0] == 0.0))

        restored, restored_ids = mesher.apply_snap_rules_at_z(
            1.0,
            return_touched_node_ids=True,
        )
        self.assertGreater(restored, 0)
        self.assertTrue(np.all(mesh.nodes[restored_ids, 0] == rail_coord))


if __name__ == "__main__":
    unittest.main()
