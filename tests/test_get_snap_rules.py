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
            mesher.get_snap_rules(2.0000005, eps=1e-6),
            mesher.get_snap_rules(2),
        )
        self.assertEqual(mesher.get_snap_rules(2.01), [])

    def test_get_restore_rules_uses_z_tolerance(self):
        """Verify lifecycle-rule lookup accepts near top-z matches."""
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
            mesher.get_restore_rules(8.0000005, eps=1e-6),
            mesher.get_restore_rules(8),
        )
        self.assertEqual(mesher.get_restore_rules(8.01), [])

    def test_high_precision_z_uses_distinct_exact_rule_buckets(self):
        """Verify raw high-precision events resolve without crossing buckets."""
        faces = [
            {
                "type": "LINE",
                "dim": [1, 0, 1, 5],
                "bottom_z": 1.23444,
                "top_z": 2.34564,
            },
            {
                "type": "LINE",
                "dim": [1.5, 6, 1.5, 11],
                "bottom_z": 1.23446,
                "top_z": 2.34566,
            },
        ]

        mesher = OptimalMesh25D()
        mesher._set_pattern(faces, element_size=5, ratio=0.2)

        first_snap_rules = mesher.get_snap_rules(1.23444)
        second_snap_rules = mesher.get_snap_rules(1.23446)
        first_restore_rules = mesher.get_restore_rules(2.34564)
        second_restore_rules = mesher.get_restore_rules(2.34566)

        self.assertEqual(
            {rule["target_coord"] for rule in first_snap_rules},
            {1.0},
        )
        self.assertEqual(
            {rule["target_coord"] for rule in second_snap_rules},
            {1.5},
        )
        self.assertEqual(
            {rule["span_min"] for rule in first_restore_rules},
            {0.0},
        )
        self.assertEqual(
            {rule["span_min"] for rule in second_restore_rules},
            {6.0},
        )

    def test_rule_getters_return_detached_copies(self):
        """Client edits to returned lifecycle metadata cannot alter a plan."""
        faces = [
            {
                "type": "LINE",
                "dim": [1.0, 0.0, 1.0, 1.0],
                "bottom_z": 0.0,
                "top_z": 1.0,
            },
            {
                "type": "LINE",
                "dim": [1.5, 0.0, 1.5, 1.0],
                "bottom_z": 2.0,
                "top_z": 3.0,
            },
        ]
        mesher = OptimalMesh25D()
        mesher._set_pattern(
            faces,
            element_size=1.0,
            ratio=1.0,
            mesh_domain={"type": "BOX", "dim": [0.0, 0.0, 3.0, 1.0]},
        )

        returned_snap = mesher.get_snap_rules(0.0)
        returned_restore = mesher.get_restore_rules(1.0)
        returned_snap[0]["target_coord"] = 99.0
        returned_restore[0]["span_min"] = 99.0

        self.assertEqual(
            mesher.snap_rules_by_z[0.0][0]["target_coord"],
            1.0,
        )
        self.assertEqual(
            mesher.restore_rules_by_z[1.0][0]["span_min"],
            0.0,
        )

    def test_direct_rule_plan_mutation_is_rejected(self):
        """Bypassing getters cannot silently change a validated snap plan."""
        faces = [
            {
                "type": "LINE",
                "dim": [1.0, 0.0, 1.0, 1.0],
                "bottom_z": 0.0,
                "top_z": 1.0,
            },
            {
                "type": "LINE",
                "dim": [1.5, 0.0, 1.5, 1.0],
                "bottom_z": 2.0,
                "top_z": 3.0,
            },
        ]
        mesher = OptimalMesh25D()
        mesher._set_pattern(
            faces,
            element_size=1.0,
            ratio=1.0,
            mesh_domain={"type": "BOX", "dim": [0.0, 0.0, 3.0, 1.0]},
        )
        mesher.mesh_checkerboard()
        mesher.snap_rules_by_z[0.0][0]["target_coord"] = 99.0

        with self.assertRaisesRegex(RuntimeError, "snap rules changed"):
            mesher.apply_snap_rules_at_z(0.0)


if __name__ == "__main__":
    unittest.main()
