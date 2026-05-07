import os
import sys
import unittest

SRC_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "src")
)
sys.path.insert(0, SRC_ROOT)

from optimal_checkerboard import OptimalMesh25D


class TestGetSnapRules(unittest.TestCase):
    def test_get_snap_rules_before_pattern(self):
        """Verify snap-rule lookup is empty before pattern setup."""
        mesher = OptimalMesh25D()

        self.assertEqual(mesher.get_snap_rules(), {})
        self.assertEqual(mesher.get_snap_rules(0), [])
        self.assertEqual(mesher.get_restore_rules(), {})
        self.assertEqual(mesher.get_restore_rules(0), [])

    def test_get_snap_rules_uses_z_tolerance(self):
        """Verify z lookup accepts near matches and rejects distant values."""
        faces = [
            {
                "type": "LINE",
                "dim": [1, 0, 1, 10],
                "bottom_z": 2,
                "top_z": 2,
            }
        ]

        mesher = OptimalMesh25D()
        mesher._set_pattern(faces, element_size=10, ratio=0.1)

        self.assertEqual(
            mesher.get_snap_rules(2.0000005),
            mesher.get_snap_rules(2),
        )
        self.assertEqual(mesher.get_snap_rules(2.01), [])

    def test_get_restore_rules_uses_z_tolerance(self):
        """Verify delayed restore-rule lookup accepts near top-z matches."""
        faces = [
            {
                "type": "LINE",
                "dim": [1, 0, 1, 10],
                "bottom_z": 2,
                "top_z": 8,
            }
        ]

        mesher = OptimalMesh25D()
        mesher._set_pattern(faces, element_size=10, ratio=0.1)

        self.assertEqual(
            mesher.get_restore_rules(8.0000005),
            mesher.get_restore_rules(8),
        )
        self.assertEqual(mesher.get_restore_rules(8.01), [])


if __name__ == "__main__":
    unittest.main()
