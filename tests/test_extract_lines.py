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
        self.assertIn(line_face, lines)
        self.assertIn([[0, 0, 0], [10, 0, 0]], lines)
        self.assertIn([[30, 5, 1], [30, 0, 1]], lines)

    def test_unsupported_face_type_error(self):
        """Verify unsupported face dictionaries fail loudly."""
        with self.assertRaisesRegex(ValueError, "Unsupported face type"):
            _extract_lines([{"type": "CIRCLE", "dim": [0, 0, 1]}])


if __name__ == "__main__":
    unittest.main()
