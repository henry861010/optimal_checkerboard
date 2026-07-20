import os
import sys
import unittest

SRC_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "src")
)
sys.path.insert(0, SRC_ROOT)

from optimal_checkerboard.algorithms.feature_lines import (
    _get_feature_lines,
    static_feature_span_endpoint_coordinates,
)


class TestStaticFeatureCoordinates(unittest.TestCase):
    def test_closed_boxes_keep_dynamic_corners_and_legal_sharing(self):
        """Verify closed geometry endpoints do not pin every feature rail."""
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

        static_coords = static_feature_span_endpoint_coordinates(faces)
        details = _get_feature_lines(
            faces,
            element_size=2.2,
            return_details=True,
            pinned_coords=static_coords,
        )

        self.assertEqual(static_coords, {"x": [], "y": []})
        self.assertEqual(details[2], [1.0, 9.0])
        self.assertEqual(details[3], [0.5, 5.0, 10.0])

    def test_isolated_line_endpoints_are_static_coordinates(self):
        """Verify finite LINE endpoints receive cross-axis mesh stations."""
        faces = [
            {
                "type": "LINE",
                "dim": [1, 2, 1, 3],
                "bottom_z": 0,
                "top_z": 10,
            },
        ]

        self.assertEqual(
            static_feature_span_endpoint_coordinates(faces),
            {"x": [], "y": [2.0, 3.0]},
        )

    def test_partial_z_coverage_keeps_endpoint_static(self):
        """Verify a perpendicular feature must cover the full z lifecycle."""
        faces = [
            {
                "type": "LINE",
                "dim": [1, 0, 1, 10],
                "bottom_z": 0,
                "top_z": 10,
            },
            {
                "type": "LINE",
                "dim": [0, 0, 2, 0],
                "bottom_z": 0,
                "top_z": 5,
            },
            {
                "type": "LINE",
                "dim": [0, 10, 2, 10],
                "bottom_z": 0,
                "top_z": 10,
            },
        ]

        static_coords = static_feature_span_endpoint_coordinates(faces)

        self.assertIn(0.0, static_coords["y"])
        self.assertNotIn(10.0, static_coords["y"])

    def test_touching_z_intervals_can_jointly_cover_endpoint(self):
        """Verify several closed intervals may form complete z coverage."""
        faces = [
            {
                "type": "LINE",
                "dim": [1, 0, 1, 10],
                "bottom_z": 0,
                "top_z": 10,
            },
            {
                "type": "LINE",
                "dim": [0, 0, 2, 0],
                "bottom_z": 0,
                "top_z": 5,
            },
            {
                "type": "LINE",
                "dim": [0, 0, 2, 0],
                "bottom_z": 5,
                "top_z": 10,
            },
        ]

        static_coords = static_feature_span_endpoint_coordinates(faces)

        self.assertNotIn(0.0, static_coords["y"])
        self.assertIn(10.0, static_coords["y"])


if __name__ == "__main__":
    unittest.main()
