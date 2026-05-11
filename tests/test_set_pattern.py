import os
import sys
import unittest

SRC_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "src")
)
sys.path.insert(0, SRC_ROOT)

from optimal_checkerboard import OptimalMesh25D
from optimal_checkerboard.data_structure.face import Face
from optimal_checkerboard.data_structure.geometry import Obj
from optimal_checkerboard.data_structure.mesh import Mesh
from optimal_checkerboard.data_structure.metal import Metal


class TestSetPattern(unittest.TestCase):
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
