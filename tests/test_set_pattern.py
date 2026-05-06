import os
import sys
import unittest

SRC_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "src")
)
sys.path.insert(0, SRC_ROOT)

from optimal_checkerboard import OptimalMesh25D


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
        group_lines_v, group_lines_h, x_list, y_list = mesher.set_pattern(
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
            mesher.set_pattern(faces, element_size=10, ratio=0.1)

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
            mesher.set_pattern(faces, element_size=10, ratio=0.1)

    def test_set_pattern_accepts_new_2d_face_schema(self):
        """Verify set_pattern accepts flat 2D dims with bottom/top z."""
        faces = [
            {
                "type": "BOX",
                "dim": [0, 0, 100, 100],
                "bottom_z": 0,
                "top_z": 100,
            },
            {
                "type": "POLYGON",
                "dim": [[10, 10], [90, 10], [90, 90], [10, 90]],
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
        _, _, x_list, y_list = mesher.set_pattern(
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


if __name__ == "__main__":
    unittest.main()
