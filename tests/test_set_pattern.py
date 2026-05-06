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
            {"type": "LINE", "dim": [[1, 0, 0], [1, 10, 0]]},
            {"type": "LINE", "dim": [[0, 3, 0], [5, 3, 0]]},
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

    def test_line_faces_reject_z_mismatch(self):
        """Verify LINE faces must stay on one z plane."""
        faces = [
            {"type": "LINE", "dim": [[1, 0, 0], [1, 10, 1]]},
        ]

        mesher = OptimalMesh25D()
        with self.assertRaisesRegex(ValueError, "Z coordinates mismatch"):
            mesher.set_pattern(faces, element_size=10, ratio=0.1)

    def test_line_faces_accept_explicit_z_range(self):
        """Verify 2D LINE faces can declare an active z interval."""
        faces = [
            {
                "type": "LINE",
                "dim": [[0, 0], [10, 0]],
                "z_range": [5, 20],
            }
        ]

        mesher = OptimalMesh25D()
        _, _, _, y_list = mesher.set_pattern(
            faces,
            element_size=10,
            ratio=0.1,
        )

        self.assertEqual(y_list, [0.0])
        self.assertEqual(mesher.get_snap_rules(5)[0]["z_top"], 20)


if __name__ == "__main__":
    unittest.main()
