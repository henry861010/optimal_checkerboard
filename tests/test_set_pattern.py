import os
import sys
import unittest

import numpy as np

SRC_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "src")
)
sys.path.insert(0, SRC_ROOT)

from optimal_checkerboard import OptimalMesh25D
from optimal_checkerboard.algorithms.snap_faces import (
    snap_faces_to_shared_rails,
)
from optimal_checkerboard.data_structure.face import Face
from optimal_checkerboard.data_structure.geometry import Obj
from optimal_checkerboard.data_structure.mesh import Mesh
from optimal_checkerboard.data_structure.metal import Metal


class TestSetPattern(unittest.TestCase):
    def test_unsafe_two_axis_sharing_falls_back_to_exact_rails(self):
        """Optimization must be abandoned before a coupled corner collapses."""
        faces = [
            {"type": "BOX", "dim": [3.5, 2, 4, 5], "bottom_z": 8, "top_z": 10},
            {"type": "BOX", "dim": [8, 7.5, 9, 8], "bottom_z": 2, "top_z": 3},
            {"type": "BOX", "dim": [1, 5.5, 2, 6], "bottom_z": 4, "top_z": 6},
            {"type": "BOX", "dim": [0, 5.5, 3, 7], "bottom_z": 0, "top_z": 3},
        ]
        mesher = OptimalMesh25D()
        mesher._set_pattern(
            faces,
            element_size=2.0,
            ratio=0.6,
            mesh_domain={"type": "BOX", "dim": [-1, -1, 13, 13]},
        )

        self.assertIsNotNone(mesher.rail_optimization_fallback)
        self.assertTrue(mesher._rail_plan_is_exact)
        mesh = mesher.mesh_checkerboard()
        baseline = mesh.nodes.copy()
        for z_value in mesher._rule_event_z_values:
            self.assertEqual(mesher.apply_snap_rules_at_z(float(z_value)), 0)
        np.testing.assert_array_equal(mesh.nodes, baseline)

    def test_filler_station_collision_triggers_exact_rail_fallback(self):
        """A moving rail may not cross a fixed densification grid line."""
        faces = [
            {
                "type": "LINE",
                "dim": [1.0, 0.0, 1.0, 1.0],
                "bottom_z": 0.0,
                "top_z": 0.0,
            },
            {
                "type": "LINE",
                "dim": [3.0, 0.0, 3.0, 1.0],
                "bottom_z": 2.0,
                "top_z": 2.0,
            },
        ]
        mesher = OptimalMesh25D()
        mesher._set_pattern(
            faces,
            element_size=1.0,
            ratio=2.0,
            mesh_domain={"type": "BOX", "dim": [0.0, 0.0, 10.0, 1.0]},
        )
        self.assertIsNone(mesher.rail_optimization_fallback)

        mesher.mesh_checkerboard()

        self.assertIsNotNone(mesher.rail_optimization_fallback)
        self.assertEqual(mesher.x_list, [1.0, 3.0])

    def test_line_faces_create_rails_and_snap_rules(self):
        """Verify standalone LINE faces enter the shared-rail pipeline."""
        faces = [
            {
                "type": "LINE",
                "dim": [1, 0, 1, 10],
                "bottom_z": 0,
                "top_z": 0,
            },
            {
                "type": "LINE",
                "dim": [0, 3, 5, 3],
                "bottom_z": 0,
                "top_z": 0,
            },
        ]

        mesher = OptimalMesh25D()
        group_lines_v, group_lines_h, x_list, y_list = mesher._set_pattern(
            faces,
            element_size=10,
            ratio=0.1,
        )

        self.assertEqual(x_list, [1.0])
        self.assertEqual(y_list, [3.0])
        self.assertEqual(len(group_lines_v), 1)
        self.assertEqual(len(group_lines_h), 1)
        self.assertEqual(
            {rule["axis"] for rule in mesher.get_snap_rules(0)},
            {"x", "y"},
        )

    def test_line_faces_require_bottom_and_top_z(self):
        """Verify LINE faces must declare the active z interval."""
        faces = [
            {"type": "LINE", "dim": [1, 0, 1, 10]},
        ]

        mesher = OptimalMesh25D()
        with self.assertRaisesRegex(ValueError, "bottom_z and top_z"):
            mesher._set_pattern(faces, element_size=10, ratio=0.1)

    def test_line_faces_reject_bad_z_range(self):
        """Verify face z intervals must be ordered."""
        faces = [
            {
                "type": "LINE",
                "dim": [0, 0, 10, 0],
                "bottom_z": 20,
                "top_z": 5,
            }
        ]

        mesher = OptimalMesh25D()
        with self.assertRaisesRegex(ValueError, "z_bottom <= z_top"):
            mesher._set_pattern(faces, element_size=10, ratio=0.1)

    def test_private_set_pattern_accepts_new_2d_face_schema(self):
        """Verify _set_pattern accepts flat 2D dims with bottom/top z."""
        faces = [
            {
                "type": "BOX",
                "dim": [0, 0, 100, 100],
                "bottom_z": 0,
                "top_z": 100,
            },
            {
                "type": "POLYGON",
                "dim": [[[10, 10], [10, 90], [90, 90], [90, 10]]],
                "bottom_z": 10,
                "top_z": 90,
            },
            {
                "type": "LINE",
                "dim": [20, 50, 80, 50],
                "bottom_z": 20,
                "top_z": 80,
            },
        ]

        mesher = OptimalMesh25D()
        _, _, x_list, y_list = mesher._set_pattern(
            faces,
            element_size=100,
            ratio=0.1,
        )

        self.assertIn(0.0, x_list)
        self.assertIn(10.0, x_list)
        self.assertIn(0.0, y_list)
        self.assertIn(10.0, y_list)
        self.assertIn(50.0, y_list)
        self.assertEqual(mesher.get_snap_rules(10)[0]["z_top"], 90)

    def test_set_pattern_obj_collects_obj_metal_mesh_and_child_faces(self):
        """Verify Obj conversion covers all pattern-bearing structures."""
        obj = Obj("BOX", [0, 0, 10, 10], z=1)
        obj.add_layer(thk=4, material="BASE")
        obj.metals.append(
            Metal(
                "NORMAL",
                begin=1,
                end=3,
                material="M1",
                ranges=[Face("BOX", [2, 2, 4, 4])],
                holes=[Face("BOX", [6, 6, 8, 8])],
            )
        )
        obj.meshs.append(Mesh(begin=0, end=4, line=[[5, 0], [5, 10]]))
        obj.meshs.append(Mesh(begin=2, end=4, face=Face("BOX", [1, 1, 3, 3])))

        child = Obj("BOX", [20, 0, 25, 5], z=2)
        child.add_layer(thk=2, material="CHILD")
        obj.add_child(child)

        mesher = OptimalMesh25D()
        mesher.set_pattern_obj(obj, element_size=10, ratio=0.1)

        self.assertIn(5.0, mesher.x_list)
        self.assertIn(20.0, mesher.x_list)
        self.assertIn(25.0, mesher.x_list)
        self.assertIn(6.0, mesher.y_list)
        self.assertIn(8.0, mesher.y_list)
        self.assertIn(2.0, mesher.get_snap_rules())
        self.assertIn(3.0, mesher.get_snap_rules())

    def test_set_pattern_obj_ignores_metal_without_ranges_or_holes(self):
        """Verify inherited metal footprint does not duplicate parent faces."""
        obj = Obj("BOX", [0, 0, 10, 10], z=0)
        obj.add_layer(thk=10, material="BASE")
        obj.metals.append(
            Metal("NORMAL", begin=2, end=8, material="M1")
        )

        mesher = OptimalMesh25D()
        mesher.set_pattern_obj(obj, element_size=10, ratio=0.1)

        self.assertEqual(sorted(mesher.get_snap_rules()), [0.0])

    def test_set_pattern_obj_warns_and_ignores_cylinder_faces(self):
        """Verify CYLINDER geometry does not enter the pattern mesher."""
        obj = Obj("CYLINDER", [0, 0, 1], z=0)
        obj.add_layer(thk=1, material="BASE")

        mesher = OptimalMesh25D()
        with self.assertWarnsRegex(RuntimeWarning, "CYLINDER"):
            mesher.set_pattern_obj(obj, element_size=10, ratio=0.1)

        self.assertEqual(mesher.x_list, [])
        self.assertEqual(mesher.y_list, [])
        self.assertEqual(mesher.get_snap_rules(), {})

    def test_get_snap_faces_snaps_lines_to_shared_rail_defaults(self):
        """Verify custom mesh inputs can use line faces on shared rails."""
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

        snap_faces = mesher.get_snap_faces()

        self.assertEqual(snap_faces[0]["dim"], [1.25, 0.0, 1.25, 5.0])
        self.assertEqual(snap_faces[1]["dim"], [1.25, 20.0, 1.25, 25.0])
        self.assertEqual(faces[0]["dim"], [1, 0, 1, 5])
        self.assertEqual(mesher.faces[1]["dim"], [1.5, 20, 1.5, 25])

    def test_snap_faces_never_matches_a_distinct_nearby_target(self):
        """Lookup tolerance must not replace one exact pattern coordinate."""
        face = {
            "type": "LINE",
            "dim": [5e-7, 0.0, 5e-7, 1.0],
            "bottom_z": 0.0,
            "top_z": 1.0,
        }
        rules = {
            0.0: [
                {
                    "axis": "x",
                    "target_coord": 0.0,
                    "rail_coord": -1.0,
                    "span_min": 0.0,
                    "span_max": 1.0,
                    "z_bottom": 0.0,
                    "z_top": 1.0,
                }
            ]
        }

        snapped = snap_faces_to_shared_rails([face], rules, eps=1e-6)

        self.assertEqual(snapped[0]["dim"], face["dim"])

    def test_get_snap_faces_keeps_rules_for_distinct_top_z_values(self):
        """Verify rule dedup retains coincident faces with different lifetimes."""
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
        mesher._set_pattern(faces, element_size=10, ratio=0.2)

        snap_faces = mesher.get_snap_faces()

        self.assertEqual(snap_faces[0]["dim"], [0.5, 0.0, 0.5, 10.0])
        self.assertEqual(snap_faces[1]["dim"], [0.5, 0.0, 0.5, 10.0])

    def test_get_snap_faces_snaps_box_edges_to_shared_rails(self):
        """Verify BOX coordinates are moved to rail defaults per edge."""
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
        mesher._set_pattern(faces, element_size=11, ratio=0.2)

        snap_faces = mesher.get_snap_faces()

        self.assertEqual(snap_faces[0]["dim"], [1.0, 0.5, 9.0, 10.0])
        self.assertEqual(snap_faces[1]["dim"], [1.0, 0.5, 9.0, 5.0])

    def test_get_snap_faces_snaps_polygon_vertices_to_shared_rails(self):
        """Verify POLYGON vertices receive both x and y rail defaults."""
        faces = [
            {
                "type": "POLYGON",
                "dim": [
                    [[0, 0], [0, 5], [5, 5], [5, 0], [0, 0]],
                ],
                "bottom_z": 0,
                "top_z": 10,
            },
            {
                "type": "POLYGON",
                "dim": [
                    [[1, 1], [1, 4], [4, 4], [4, 1], [1, 1]],
                ],
                "bottom_z": 11,
                "top_z": 20,
            },
        ]

        mesher = OptimalMesh25D()
        mesher._set_pattern(faces, element_size=10, ratio=0.2)

        snap_faces = mesher.get_snap_faces()

        self.assertEqual(
            snap_faces[0]["dim"],
            [
                [
                    [0.5, 0.5],
                    [0.5, 4.5],
                    [4.5, 4.5],
                    [4.5, 0.5],
                    [0.5, 0.5],
                ],
            ],
        )
        self.assertEqual(
            snap_faces[0]["dim"][0][0],
            snap_faces[0]["dim"][0][-1],
        )
        self.assertEqual(
            snap_faces[1]["dim"],
            [
                [
                    [0.5, 0.5],
                    [0.5, 4.5],
                    [4.5, 4.5],
                    [4.5, 0.5],
                    [0.5, 0.5],
                ],
            ],
        )


if __name__ == "__main__":
    unittest.main()
