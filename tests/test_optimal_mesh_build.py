import os
import sys
import unittest

SRC_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "src")
)
sys.path.insert(0, SRC_ROOT)

import numpy as np

from optimal_checkerboard import OptimalMesh25D
from optimal_checkerboard.algorithms.drag import Dragger


def _area():
    return {
        "type": "BOX",
        "dim": [0, 0, 3, 3],
        "material": "CORE",
    }


def _build_zero_copy_mesher():
    faces = [
        {
            "type": "LINE",
            "dim": [1, 0, 1, 3],
            "bottom_z": 0,
            "top_z": 10,
        },
        {
            "type": "LINE",
            "dim": [1.5, 0, 1.5, 3],
            "bottom_z": 20,
            "top_z": 30,
        },
    ]

    mesher = OptimalMesh25D()
    mesher._set_pattern(
        faces,
        element_size=5,
        ratio=0.2,
        mesh_domain={"type": "BOX", "dim": [0, 0, 3, 3]},
    )
    mesher.mesh_checkerboard()
    return mesher


class TestOptimalMeshBuild(unittest.TestCase):
    def test_build_returns_dragger_and_mutates_mesh2d_by_default(self):
        mesher = _build_zero_copy_mesher()
        obj_list = [
            [
                {"z": 0, "areas": [_area()], "element_size": 10},
                {"z": 10},
            ]
        ]

        dragger = mesher.build(obj_list)

        self.assertIsInstance(dragger, Dragger)
        self.assertIs(mesher.dragger, dragger)
        self.assertTrue(np.shares_memory(dragger.node_2D, mesher.mesh2d.nodes))

        x_values = {
            round(float(x), 6)
            for x in mesher.mesh2d.nodes[:, 0]
        }
        self.assertIn(1.0, x_values)
        self.assertNotIn(1.25, x_values)

    def test_build_can_preserve_mesh2d_with_one_node_copy(self):
        mesher = _build_zero_copy_mesher()
        before = mesher.mesh2d.nodes.copy()
        obj_list = [
            [
                {"z": 0, "areas": [_area()], "element_size": 10},
                {"z": 10},
            ]
        ]

        dragger = mesher.build(obj_list, preserve_mesh2d=True)

        self.assertFalse(np.shares_memory(dragger.node_2D, mesher.mesh2d.nodes))
        np.testing.assert_allclose(mesher.mesh2d.nodes, before)
        x_values = {
            round(float(x), 6)
            for x in dragger.node_2D[:, 0]
        }
        self.assertIn(1.0, x_values)
        self.assertNotIn(1.25, x_values)

    def test_build_syncs_snapped_top_layer_3d_nodes(self):
        faces = [
            {
                "type": "LINE",
                "dim": [1, 0, 1, 3],
                "bottom_z": 0,
                "top_z": 10,
            },
            {
                "type": "LINE",
                "dim": [1.5, 0, 1.5, 3],
                "bottom_z": 11,
                "top_z": 20,
            },
        ]
        mesher = OptimalMesh25D()
        mesher._set_pattern(
            faces,
            element_size=5,
            ratio=0.2,
            mesh_domain={"type": "BOX", "dim": [0, 0, 3, 3]},
        )
        mesher.mesh_checkerboard()
        area = _area()
        obj_list = [
            [
                {"z": 0, "areas": [area], "element_size": 10},
                {"z": 10, "areas": [area], "element_size": 1},
                {"z": 11, "areas": [area], "element_size": 10},
                {"z": 20},
            ]
        ]

        dragger = mesher.build(obj_list)

        nodes = dragger.nodes[:dragger.node_num]
        z11_nodes = nodes[np.isclose(nodes[:, 2], 11)]
        z11_x_values = {
            round(float(x), 6)
            for x in z11_nodes[:, 0]
        }
        self.assertIn(1.5, z11_x_values)
        self.assertNotIn(1.0, z11_x_values)
        self.assertNotIn(1.25, z11_x_values)

    def test_build_applies_distinct_raw_high_precision_z_events(self):
        """Verify build resolves raw z values to separate rounded buckets."""
        first_z = 1.23444
        second_z = 1.23446
        faces = [
            {
                "type": "LINE",
                "dim": [1, 0, 1, 5],
                "bottom_z": first_z,
                "top_z": second_z,
            },
            {
                "type": "LINE",
                "dim": [1.5, 6, 1.5, 11],
                "bottom_z": second_z,
                "top_z": second_z + 1,
            },
        ]
        mesher = OptimalMesh25D()
        mesher._set_pattern(
            faces,
            element_size=5,
            ratio=0.2,
            mesh_domain={"type": "BOX", "dim": [0, 0, 3, 11]},
        )
        mesher.mesh_checkerboard()
        area = {"type": "BOX", "dim": [0, 0, 3, 11], "material": "CORE"}

        dragger = mesher.build(
            [[
                {"z": first_z, "areas": [area], "element_size": 1},
                {"z": second_z, "areas": [area], "element_size": 1},
                {"z": second_z + 1},
            ]],
            preserve_mesh2d=True,
        )
        nodes = dragger.nodes[:dragger.node_num]
        first_plane = nodes[np.isclose(nodes[:, 2], first_z, atol=1e-9)]
        second_plane = nodes[np.isclose(nodes[:, 2], second_z, atol=1e-9)]

        self.assertTrue(
            np.any(
                np.isclose(first_plane[:, 0], 1.0)
                & (first_plane[:, 1] <= 5)
            )
        )
        self.assertTrue(
            np.any(
                np.isclose(second_plane[:, 0], 1.5)
                & (second_plane[:, 1] >= 6)
            )
        )

    def test_each_object_stack_starts_from_shared_rail_baseline(self):
        """Verify stacked traversals restore coordinates and element areas."""
        faces = [
            {
                "type": "LINE",
                "dim": [1, 0, 1, 3],
                "bottom_z": 10,
                "top_z": 20,
            },
            {
                "type": "LINE",
                "dim": [1.5, 0, 1.5, 3],
                "bottom_z": 30,
                "top_z": 40,
            },
        ]
        mesher = OptimalMesh25D()
        mesher._set_pattern(
            faces,
            element_size=5,
            ratio=0.2,
            mesh_domain={"type": "BOX", "dim": [0, 0, 3, 3]},
        )
        mesher.mesh_checkerboard()
        area = _area()
        first_stack = [
            {"z": z, "areas": [area], "element_size": 10}
            for z in (0, 10, 20, 30)
        ] + [{"z": 40}]
        metal = {
            "type": "NORMAL",
            "material": "M1",
            "density": 0,
            "ranges": [{"type": "BOX", "dim": [0, 0, 1.25, 3]}],
        }
        second_area = dict(area)
        second_area["metals"] = [metal]
        second_stack = [
            {"z": 0, "areas": [second_area], "element_size": 10},
            {"z": 5},
        ]

        dragger = mesher.build(
            [first_stack, second_stack],
            preserve_mesh2d=True,
        )
        nodes = dragger.nodes[:dragger.node_num]
        z0_x_values, counts = np.unique(
            np.round(nodes[np.isclose(nodes[:, 2], 0), 0], 6),
            return_counts=True,
        )
        z0_counts = dict(zip(z0_x_values, counts))

        self.assertEqual(z0_counts[1.25], 4)
        self.assertNotIn(1.5, z0_counts)
        self.assertAlmostEqual(metal["volumn"], 3.75)


if __name__ == "__main__":
    unittest.main()
