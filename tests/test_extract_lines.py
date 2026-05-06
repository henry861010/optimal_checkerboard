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
        line_face = [[20, 1, 0], [25, 1, 0]]
        faces = [
            {"type": "BOX", "dim": [0, 0, 0, 10, 5, 3]},
            {"type": "LINE", "dim": line_face},
            {
                "type": "POLYGON",
                "dim": [
                    [
                        [30, 0, 1],
                        [35, 0, 1],
                        [35, 5, 1],
                        [30, 5, 1],
                    ]
                ],
            },
        ]

        lines = _extract_lines(faces)

        self.assertEqual(len(lines), 9)
        self.assertIn([[20.0, 1.0], [25.0, 1.0], [0.0, 0.0]], lines)
        self.assertIn([[0.0, 0.0], [10.0, 0.0], [0.0, 0.0]], lines)
        self.assertIn([[30.0, 5.0], [30.0, 0.0], [1.0, 1.0]], lines)

    def test_extract_faces_with_explicit_z_range(self):
        """Verify xy lines can carry an explicit z interval."""
        faces = [
            {
                "type": "LINE",
                "dim": [[0, 0], [10, 0]],
                "z_range": [2, 8],
            },
            {
                "type": "BOX",
                "dim": [20, 0, 30, 5],
                "z_bottom": 3,
                "z_top": 7,
            },
        ]

        lines = _extract_lines(faces)

        self.assertIn([[0.0, 0.0], [10.0, 0.0], [2.0, 8.0]], lines)
        self.assertIn([[20.0, 0.0], [30.0, 0.0], [3.0, 7.0]], lines)

    def test_unsupported_face_type_error(self):
        """Verify unsupported face dictionaries fail loudly."""
        with self.assertRaisesRegex(ValueError, "Unsupported face type"):
            _extract_lines([{"type": "CIRCLE", "dim": [0, 0, 1]}])


if __name__ == "__main__":
    unittest.main()
