import os
import sys
import unittest

SRC_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "src")
)
sys.path.insert(0, SRC_ROOT)

from optimal_checkerboard.algorithms.feature_lines import _get_feature_lines


class TestGetFeatureLines(unittest.TestCase):
    def test_get_feature_lines_return_details(self):
        """Verify detailed feature extraction returns rails and snap rules."""
        faces = [
            {
                "type": "LINE",
                "dim": [1, 0, 1, 5],
                "bottom_z": 0,
                "top_z": 0,
            },
            {
                "type": "LINE",
                "dim": [1.5, 6, 1.5, 11],
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

        (
            group_lines_v,
            group_lines_h,
            x_list,
            y_list,
            snap_rules_by_z,
            restore_rules_by_z,
            rails,
        ) = _get_feature_lines(faces, element_size=1.0, return_details=True)

        self.assertEqual(x_list, [1.25])
        self.assertEqual(y_list, [3.0])
        self.assertEqual(len(group_lines_v), 1)
        self.assertEqual(len(group_lines_h), 1)
        self.assertEqual(set(rails), {"x", "y"})
        self.assertEqual(len(snap_rules_by_z[0.0]), 3)
        self.assertEqual(len(restore_rules_by_z[0.0]), 3)


if __name__ == "__main__":
    unittest.main()
