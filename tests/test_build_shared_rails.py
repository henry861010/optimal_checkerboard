import os
import sys
import unittest
from unittest import mock

SRC_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "src")
)
sys.path.insert(0, SRC_ROOT)

from optimal_checkerboard.algorithms import rail_builder as rail_builder_module
from optimal_checkerboard.algorithms.rail_builder import build_shared_rails


class TestBuildSharedRails(unittest.TestCase):
    def test_thousand_adversarial_features_do_not_sort_per_candidate(self):
        """Guard the geometry<=1,000 rail-order preprocessing bound."""
        feature_count = 1000
        lines = [
            [
                [index / feature_count, 2.0 * index],
                [index / feature_count, 2.0 * index + 1.0],
                [0.0, 1.0],
            ]
            for index in range(feature_count)
        ]
        original = rail_builder_module._rails_with_virtual_pins

        with mock.patch.object(
            rail_builder_module,
            "_rails_with_virtual_pins",
            wraps=original,
        ) as full_sort:
            result = build_shared_rails(lines, merge_tol=2.0)

        self.assertEqual(len(result["x_rails"]), feature_count)
        self.assertEqual(
            sum(len(rules) for rules in result["snap_rules_by_z"].values()),
            feature_count,
        )
        # One final global-order proof is expected.  The former hot path
        # called this for O(feature_count**2) greedy candidates.
        self.assertLessEqual(full_sort.call_count, 2)

    def test_same_z_non_overlapping_lines_share_one_rail(self):
        """Verify same-z non-overlapping lines share one checkerboard rail."""
        lines = [
            [[1, 0, 0], [1, 10, 0]],
            [[1.5, 12, 0], [1.5, 20, 0]],
        ]

        result = build_shared_rails(lines, merge_tol=1.0)

        self.assertEqual(result["x_list"], [1.25])
        self.assertEqual(len(result["snap_rules_by_z"][0.0]), 2)
        self.assertEqual(len(result["restore_rules_by_z"][0.0]), 2)

    def test_targets_closer_than_eps_remain_distinct_when_spans_overlap(self):
        """Verify numeric tolerance never substitutes for target identity."""
        lines = [
            [[1.0, 0], [1.0, 10], [0, 10]],
            [[1.005, 0], [1.005, 10], [0, 10]],
        ]

        result = build_shared_rails(lines, merge_tol=1.0, eps=0.01)

        self.assertEqual(result["x_list"], [1.0, 1.005])
        target_coords = sorted(
            rule["target_coord"]
            for rule in result["snap_rules_by_z"][0.0]
        )
        self.assertEqual(target_coords, [1.0, 1.005])

    def test_zero_merge_tolerance_only_groups_exact_coordinates(self):
        """Verify ratio zero cannot inherit a fixed numeric merge window."""
        lines = [
            [[1.0, 0], [1.0, 10], [0, 10]],
            [[1.005, 20], [1.005, 30], [0, 10]],
        ]

        result = build_shared_rails(lines, merge_tol=0.0)

        self.assertEqual(result["x_list"], [1.0, 1.005])

    def test_near_diagonal_line_is_not_projected_to_an_axis(self):
        """Verify scale-aware noise handling rejects real diagonal geometry."""
        lines = [
            [[1.0, 0], [1.005, 10], [0, 10]],
        ]

        with self.assertRaisesRegex(ValueError, "strictly vertical"):
            build_shared_rails(lines, merge_tol=1.0)

    def test_close_z_events_keep_exact_distinct_rule_buckets(self):
        """Verify z topology is never rounded into a shared event key."""
        first_z = 1.00001
        second_z = 1.00002
        lines = [
            [[1.0, 0], [1.0, 10], [first_z, first_z]],
            [[1.5, 0], [1.5, 10], [second_z, second_z]],
        ]

        result = build_shared_rails(lines, merge_tol=1.0)

        self.assertEqual(
            list(result["snap_rules_by_z"]),
            [first_z, second_z],
        )
        self.assertEqual(result["x_list"], [1.25])

    def test_new_rail_triggers_final_global_order_repair(self):
        """Verify a late singleton cannot touch a shared rail target."""
        lines = [
            [[41, 0], [41, 10], [-1, 10]],
            [[47, 20], [47, 30], [-1, 10]],
            [[47, 0], [47, 10], [0, 10]],
        ]

        result = build_shared_rails(lines, merge_tol=6.0)

        self.assertEqual(result["x_list"], [41.0, 47.0])
        self.assertEqual(
            [rail["member_count"] for rail in result["x_rails"]],
            [1, 2],
        )

    def test_safe_fallback_is_independent_of_feature_input_order(self):
        """Verify conservative regrouping has deterministic public output."""
        lines = [
            [[41, 0], [41, 10], [-1, 10]],
            [[47, 20], [47, 30], [-1, 10]],
            [[47, 0], [47, 10], [0, 10]],
        ]
        baseline = build_shared_rails(lines, merge_tol=6.0)

        for reordered in (list(reversed(lines)), lines[1:] + lines[:1]):
            self.assertEqual(
                build_shared_rails(reordered, merge_tol=6.0),
                baseline,
            )

    def test_diagonal_boxes_at_merge_boundary_keep_distinct_corners(self):
        """Verify independent x/y grouping cannot collapse diagonal corners."""
        lines = [
            [[5, 5], [5, 20], [0, 10]],
            [[20, 5], [20, 20], [0, 10]],
            [[5, 5], [20, 5], [0, 10]],
            [[5, 20], [20, 20], [0, 10]],
            [[21, 21], [21, 35], [0, 10]],
            [[35, 21], [35, 35], [0, 10]],
            [[21, 21], [35, 21], [0, 10]],
            [[21, 35], [35, 35], [0, 10]],
        ]

        result = build_shared_rails(lines, merge_tol=1.0)

        self.assertEqual(result["x_list"], [5.0, 20.0, 21.0, 35.0])
        self.assertEqual(result["y_list"], [5.0, 20.0, 21.0, 35.0])

    def test_pinned_coordinates_are_structural_ordering_barriers(self):
        """Verify domain coordinates cannot be displaced by rail sharing."""
        lines = [
            [[0, 0], [0, 10], [0, 10]],
            [[1, 20], [1, 30], [0, 10]],
        ]

        result = build_shared_rails(
            lines,
            merge_tol=1.0,
            pinned_coords={"x": [0.0, 10.0], "y": [0.0, 30.0]},
        )

        self.assertEqual(result["x_list"], [0.0, 1.0])
        self.assertTrue(result["x_rails"][0]["pinned"])
        self.assertFalse(result["x_rails"][1]["pinned"])

    def test_virtual_pinned_coordinate_blocks_cross_boundary_merge(self):
        """Verify a structural coordinate need not be a pattern feature."""
        lines = [
            [[-1, 0], [-1, 10], [0, 10]],
            [[1, 20], [1, 30], [0, 10]],
        ]

        result = build_shared_rails(
            lines,
            merge_tol=2.0,
            pinned_coords={"x": [0.0]},
        )

        self.assertEqual(result["x_list"], [-1.0, 1.0])

    def test_same_z_near_corner_vertical_lines_do_not_share_rail(self):
        """Verify near endpoints stay on separate vertical rails."""
        lines = [
            [[10, 0, 0], [10, 10, 0]],
            [[11, 12, 0], [11, 18, 0]],
        ]

        result = build_shared_rails(lines, merge_tol=2.2)

        self.assertEqual(result["x_list"], [10.0, 11.0])

    def test_static_facing_endpoints_allow_safe_same_axis_sharing(self):
        """Verify pinned cross rails remove independent-corner collision."""
        lines = [
            [[1, 0], [1, 10], [0, 0]],
            [[1.5, 11], [1.5, 20], [0, 0]],
        ]

        result = build_shared_rails(
            lines,
            merge_tol=1.0,
            pinned_coords={"y": [10.0, 11.0]},
        )

        self.assertEqual(result["x_list"], [1.25])

    def test_same_z_near_corner_horizontal_lines_do_not_share_rail(self):
        """Verify near endpoints stay on separate horizontal rails."""
        lines = [
            [[0, 10, 0], [10, 10, 0]],
            [[11, 12, 0], [18, 12, 0]],
        ]

        result = build_shared_rails(lines, merge_tol=2.2)

        self.assertEqual(result["y_list"], [10.0, 12.0])

    def test_same_z_overlapping_lines_do_not_share_rail(self):
        """Verify same-z overlapping lines are assigned separate rails."""
        lines = [
            [[1, 0, 0], [1, 10, 0]],
            [[1.5, 5, 0], [1.5, 15, 0]],
        ]

        result = build_shared_rails(lines, merge_tol=1.0)

        self.assertEqual(result["x_list"], [1.0, 1.5])

    def test_same_z_touching_lines_do_not_share_rail(self):
        """Verify touching same-z spans cannot share different targets."""
        lines = [
            [[1, 0, 0], [1, 10, 0]],
            [[1.5, 10, 0], [1.5, 20, 0]],
        ]

        result = build_shared_rails(lines, merge_tol=1.0)

        self.assertEqual(result["x_list"], [1.0, 1.5])

    def test_duplicate_lines_emit_one_snap_rule(self):
        """Verify identical duplicate features do not duplicate snap rules."""
        line = [[1, 0, 0], [1, 10, 0]]

        result = build_shared_rails([line, line], merge_tol=1.0)

        self.assertEqual(result["x_list"], [1.0])
        self.assertEqual(len(result["snap_rules_by_z"][0.0]), 1)
        self.assertEqual(len(result["restore_rules_by_z"][0.0]), 1)

    def test_same_z_intermediate_overlap_blocks_rail_sharing(self):
        """Verify a conflicting intermediate line blocks cross-rail merging."""
        lines = [
            [[10, 0, 0], [10, 10, 0]],
            [[11, 5, 0], [11, 30, 0]],
            [[12, 20, 0], [12, 40, 0]],
        ]

        result = build_shared_rails(lines, merge_tol=2.2)

        self.assertEqual(result["x_list"], [10.0, 11.0, 12.0])

    def test_cross_z_overlapping_lines_can_share_rail(self):
        """Verify cross-z overlapping lines can share one rail."""
        lines = [
            [[0, 1, 0], [0, 10, 0]],
            [[1.5, 5, 10], [1.5, 15, 10]],
        ]

        result = build_shared_rails(lines, merge_tol=2.0)

        self.assertEqual(result["x_list"], [0.75])
        self.assertIn(0.0, result["snap_rules_by_z"])
        self.assertIn(10.0, result["snap_rules_by_z"])

    def test_touching_positive_z_ranges_with_conflicting_targets_do_not_share(self):
        """Verify a top/bottom event cannot reuse one rail for two targets."""
        lines = [
            [[1, 0], [1, 10], [0, 10]],
            [[1.5, 0], [1.5, 10], [10, 20]],
        ]

        result = build_shared_rails(lines, merge_tol=1.0)

        self.assertEqual(result["x_list"], [1.0, 1.5])

    def test_zero_height_event_at_positive_top_does_not_share_conflicting_rail(self):
        """Verify a legacy point event conflicts with an inclusive top plane."""
        lines = [
            [[1, 0], [1, 10], [0, 10]],
            [[1.5, 0], [1.5, 10], [10, 10]],
        ]

        result = build_shared_rails(lines, merge_tol=1.0)

        self.assertEqual(result["x_list"], [1.0, 1.5])

    def test_same_target_z_chain_restores_only_after_last_top(self):
        """Verify overlapping copies stay snapped through their full z union."""
        lines = [
            [[0, 0], [0, 10], [0, 5]],
            [[0, 0], [0, 10], [0, 10]],
            [[1, 20], [1, 30], [0, 10]],
        ]

        result = build_shared_rails(lines, merge_tol=2.0)
        target_rules = [
            rule
            for rule in result["snap_rules_by_z"][0.0]
            if rule["target_coord"] == 0.0
        ]

        self.assertEqual(result["x_list"], [0.5])
        self.assertEqual(
            sorted(rule["z_top"] for rule in target_rules),
            [5.0, 10.0],
        )
        self.assertNotIn(5.0, result["restore_rules_by_z"])
        self.assertEqual(
            len(
                [
                    rule
                    for rule in result["restore_rules_by_z"][10.0]
                    if rule["span_min"] == 0.0
                    and rule["span_max"] == 10.0
                ]
            ),
            1,
        )

    def test_overlapping_z_ranges_prevent_span_conflicting_rail_sharing(self):
        """Verify active z overlap prevents nested faces from sharing rails."""
        lines = [
            [[0, 0], [100, 0], [0, 100]],
            [[10, 10], [90, 10], [10, 90]],
        ]

        result = build_shared_rails(lines, merge_tol=20.0)

        self.assertEqual(result["y_list"], [0.0, 10.0])

    def test_same_coord_overlapping_z_ranges_can_share_rail(self):
        """Verify equal target coordinates can share through a z overlap."""
        lines = [
            [[0, 0], [100, 0], [0, 100]],
            [[10, 0], [90, 0], [10, 90]],
        ]

        result = build_shared_rails(lines, merge_tol=20.0)

        self.assertEqual(result["y_list"], [0.0])
        self.assertIn(0.0, result["snap_rules_by_z"])
        self.assertIn(10.0, result["snap_rules_by_z"])

    def test_overlapping_z_ranges_prevent_neighbor_rail_target_reversal(self):
        """Verify adjacent shared rails cannot snap into reversed x order."""
        lines = [
            [[10, 0], [10, 10], [0, 100]],
            [[14, 12], [14, 20], [0, 100]],
            [[15, 23], [15, 30], [0, 100]],
            [[20, 37], [20, 45], [0, 100]],
        ]

        result = build_shared_rails(lines, merge_tol=6.0)

        self.assertEqual(result["x_list"], [10.0, 14.0, 17.5])


if __name__ == "__main__":
    unittest.main()
