import os
import sys
import unittest

SRC_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "src")
)
sys.path.insert(0, SRC_ROOT)

from optimal_checkerboard.algorithms.feature_lines import _extract_lines


class TestExtractLines(unittest.TestCase):
    def test_extract_box_polygon_and_line_faces(self):
        """Verify all supported face types are converted into line segments."""
        line_face = [20, 1, 25, 1]
        faces = [
            {
                "type": "BOX",
                "dim": [0, 0, 10, 5],
                "bottom_z": 0,
                "top_z": 0,
            },
            {
                "type": "LINE",
                "dim": line_face,
                "bottom_z": 0,
                "top_z": 0,
            },
            {
                "type": "POLYGON",
                "dim": [
                    [
                        [30, 0],
                        [30, 5],
                        [35, 5],
                        [35, 0],
                    ],
                ],
                "bottom_z": 1,
                "top_z": 1,
            },
        ]

        lines = _extract_lines(faces)

        self.assertEqual(len(lines), 9)
        self.assertIn([[20.0, 1.0], [25.0, 1.0], [0.0, 0.0]], lines)
        self.assertIn([[0.0, 0.0], [10.0, 0.0], [0.0, 0.0]], lines)
        self.assertIn([[30.0, 0.0], [30.0, 5.0], [1.0, 1.0]], lines)

    def test_extract_faces_require_bottom_and_top_z(self):
        """Verify face inputs without bottom/top z fail loudly."""
        with self.assertRaisesRegex(ValueError, "bottom_z and top_z"):
            _extract_lines([{"type": "LINE", "dim": [0, 0, 10, 0]}])

    def test_extract_new_2d_face_schema(self):
        """Verify new BOX, POLYGON, and LINE face dims use bottom/top z."""
        faces = [
            {
                "type": "BOX",
                "dim": [0, 0, 10, 5],
                "bottom_z": 2,
                "top_z": 8,
            },
            {
                "type": "POLYGON",
                "dim": [[[20, 0], [20, 5], [25, 5], [25, 0]]],
                "bottom_z": 3,
                "top_z": 9,
            },
            {
                "type": "LINE",
                "dim": [30, 1, 35, 1],
                "bottom_z": 4,
                "top_z": 10,
            },
        ]

        lines = _extract_lines(faces)

        self.assertEqual(len(lines), 9)
        self.assertIn([[0.0, 0.0], [10.0, 0.0], [2.0, 8.0]], lines)
        self.assertIn([[20.0, 0.0], [20.0, 5.0], [3.0, 9.0]], lines)
        self.assertIn([[30.0, 1.0], [35.0, 1.0], [4.0, 10.0]], lines)

    def test_extract_polygon_hull_and_hole_loops(self):
        """Verify POLYGON loops include clockwise hulls and CCW holes."""
        faces = [
            {
                "type": "POLYGON",
                "dim": [
                    [[0, 0], [0, 5], [5, 5], [5, 0]],
                    [[2, 2], [3, 2], [3, 3], [2, 3]],
                ],
                "bottom_z": 0,
                "top_z": 10,
            },
        ]

        lines = _extract_lines(faces)

        self.assertEqual(len(lines), 8)
        self.assertIn([[0.0, 0.0], [0.0, 5.0], [0.0, 10.0]], lines)
        self.assertIn([[2.0, 2.0], [3.0, 2.0], [0.0, 10.0]], lines)

    def test_polygon_rejects_flat_legacy_dim(self):
        """Verify POLYGON dim must be nested into loops."""
        with self.assertRaisesRegex(ValueError, r"\[\[\[x1,y1\]"):
            _extract_lines(
                [
                    {
                        "type": "POLYGON",
                        "dim": [[0, 0], [0, 1], [1, 1], [1, 0]],
                        "bottom_z": 0,
                        "top_z": 1,
                    }
                ]
            )

    def test_unsupported_face_type_error(self):
        """Verify unsupported face dictionaries fail loudly."""
        with self.assertRaisesRegex(ValueError, "Unsupported face type"):
            _extract_lines([{"type": "CIRCLE", "dim": [0, 0, 1]}])


if __name__ == "__main__":
    unittest.main()
