import os
import sys
import tracemalloc
import unittest

SRC_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "src")
)
sys.path.insert(0, SRC_ROOT)

from optimal_checkerboard.algorithms.snap_plan_validation import (
    UnsafeSnapPlanError,
    validate_snap_plan,
)
from optimal_checkerboard.algorithms.feature_lines import (
    _get_feature_lines,
    static_feature_span_endpoint_coordinates,
)
from optimal_checkerboard.mesh.domain import (
    mesh_domain_pinned_coordinates,
    normalize_mesh_domain,
)
from optimal_checkerboard.mesh.generator import generate_checkerboard_mesh


def _rail(axis, rail_id, coord, pinned=False):
    return {
        "axis": axis,
        "rail_id": rail_id,
        "coord": coord,
        "pinned": pinned,
    }


def _rule(
    axis,
    rail_id,
    rail_coord,
    target,
    span_min,
    span_max,
    z_bottom=0.0,
    z_top=1.0,
):
    return {
        "axis": axis,
        "rail_id": rail_id,
        "rail_coord": rail_coord,
        "target_coord": target,
        "span_min": span_min,
        "span_max": span_max,
        "z": z_bottom,
        "z_bottom": z_bottom,
        "z_top": z_top,
    }


class TestSnapPlanValidation(unittest.TestCase):
    def test_audit_four_box_reproduction_is_unsafe(self):
        """Catch the production four-BOX rail-sharing corruption case."""
        faces = [
            {
                "type": "BOX",
                "dim": [3.5, 2.0, 4.0, 5.0],
                "bottom_z": 8.0,
                "top_z": 10.0,
            },
            {
                "type": "BOX",
                "dim": [8.0, 7.5, 9.0, 8.0],
                "bottom_z": 2.0,
                "top_z": 3.0,
            },
            {
                "type": "BOX",
                "dim": [1.0, 5.5, 2.0, 6.0],
                "bottom_z": 4.0,
                "top_z": 6.0,
            },
            {
                "type": "BOX",
                "dim": [0.0, 5.5, 3.0, 7.0],
                "bottom_z": 0.0,
                "top_z": 3.0,
            },
        ]
        element_size = 2.0
        domain = normalize_mesh_domain(
            {"type": "BOX", "dim": [-1.0, -1.0, 13.0, 13.0]}
        )
        mandatory = static_feature_span_endpoint_coordinates(faces)
        domain_pins = mesh_domain_pinned_coordinates(domain)
        pinned = {
            axis: sorted(set(mandatory[axis]) | set(domain_pins[axis]))
            for axis in ("x", "y")
        }
        details = _get_feature_lines(
            faces,
            element_size=element_size * 0.6,
            return_details=True,
            pinned_coords=pinned,
        )
        mesh_result = generate_checkerboard_mesh(
            domain,
            details[2],
            details[3],
            element_size,
            mandatory_coordinates=mandatory,
            return_metadata=True,
        )
        metadata = mesh_result[-1]

        with self.assertRaises(UnsafeSnapPlanError) as caught:
            validate_snap_plan(
                {"x": metadata["x_nodes"], "y": metadata["y_nodes"]},
                details[6],
                details[4],
            )

        failure = caught.exception.failure
        self.assertEqual(failure.reason, "non_ccw_or_non_convex_cell")
        self.assertEqual(failure.state.value, 0.0)
        self.assertIn(0.0, failure.cross_products)

    def test_safe_partial_rail_snap_checks_boundaries_and_interval(self):
        axes = {"x": [0.0, 1.0, 2.0, 3.0], "y": [0.0, 1.0, 2.0]}
        rails = {"x": [_rail("x", 0, 1.0)], "y": []}
        rules = {
            0.0: [_rule("x", 0, 1.0, 1.25, 0.0, 2.0, 0.0, 2.0)]
        }

        result = validate_snap_plan(axes, rails, rules)

        self.assertTrue(result.safe)
        self.assertEqual(result.critical_state_count, 3)
        # The exact boundaries and interior have the same active geometry.
        self.assertEqual(result.evaluated_state_count, 1)
        self.assertGreater(result.checked_cell_count, 0)

    def test_single_axis_crossing_is_unsafe(self):
        axes = {"x": [0.0, 1.0, 2.0, 3.0], "y": [0.0, 1.0]}
        rails = {"x": [_rail("x", 0, 1.0)], "y": []}
        rules = [_rule("x", 0, 1.0, 2.25, 0.0, 1.0)]

        with self.assertRaises(UnsafeSnapPlanError) as caught:
            validate_snap_plan(axes, rails, rules)

        failure = caught.exception.failure
        self.assertEqual(failure.reason, "non_ccw_or_non_convex_cell")
        self.assertEqual(failure.cell, (1, 0))
        self.assertTrue(any(value < 0.0 for value in failure.cross_products))

    def test_collision_with_non_rail_filler_station_is_unsafe(self):
        axes = {"x": [0.0, 1.0, 1.5, 3.0], "y": [0.0, 1.0, 2.0]}
        rails = {"x": [_rail("x", 7, 1.0)], "y": []}
        rules = [_rule("x", 7, 1.0, 1.5, 0.0, 2.0)]

        result = validate_snap_plan(
            axes,
            rails,
            rules,
            raise_on_unsafe=False,
        )

        self.assertFalse(result.safe)
        self.assertEqual(
            result.failure.reason,
            "non_ccw_or_non_convex_cell",
        )
        self.assertIn(0.0, result.failure.cross_products)

    def test_diagonal_box_corner_coupling_rejects_old_four_rule_plan(self):
        """Reproduce independent x/y grouping of two diagonal BOX corners."""
        axes = {
            "x": [0.0, 10.0, 10.5, 11.0, 20.0],
            "y": [0.0, 10.0, 10.5, 11.0, 20.0],
        }
        rails = {
            "x": [_rail("x", 0, 10.5)],
            "y": [_rail("y", 0, 10.5)],
        }
        rules = [
            _rule("x", 0, 10.5, 10.0, 0.0, 10.0),
            _rule("x", 0, 10.5, 11.0, 11.0, 20.0),
            _rule("y", 0, 10.5, 10.0, 0.0, 10.0),
            _rule("y", 0, 10.5, 11.0, 11.0, 20.0),
        ]

        result = validate_snap_plan(
            axes,
            rails,
            rules,
            raise_on_unsafe=False,
        )

        self.assertFalse(result.safe)
        self.assertEqual(result.failure.reason, "conflicting_node_targets")
        self.assertEqual(result.failure.node, (2, 2))
        self.assertEqual(set(result.failure.rule_indices), {0, 1})

    def test_missing_exact_span_endpoint_is_rejected(self):
        axes = {"x": [0.0, 1.0, 2.0], "y": [0.0, 0.5, 1.0]}
        rails = {"x": [_rail("x", 0, 1.0)], "y": []}
        rules = [_rule("x", 0, 1.0, 1.25, 0.25, 1.0)]

        result = validate_snap_plan(
            axes,
            rails,
            rules,
            raise_on_unsafe=False,
        )

        self.assertFalse(result.safe)
        self.assertEqual(result.failure.reason, "feature_not_represented")

    def test_work_is_sparse_in_axis_product(self):
        # A 4,000 x 4,000 structural grid has about 16 million cells.  A short
        # rule must inspect only the cells incident to its touched rail nodes.
        axis = [float(index) for index in range(4000)]
        axes = {"x": axis, "y": axis}
        rails = {"x": [_rail("x", 0, 2000.0)], "y": []}
        rules = [
            _rule("x", 0, 2000.0, 2000.25, 1000.0, 1010.0, 0.0, 2.0)
        ]

        result = validate_snap_plan(axes, rails, rules)

        self.assertTrue(result.safe)
        self.assertLess(result.checked_cell_count, 100)

    def test_noop_rule_still_detects_same_rail_moving_overlap(self):
        axes = {"x": [0.0, 1.0, 2.0], "y": [0.0, 1.0, 2.0]}
        rails = {"x": [_rail("x", 0, 1.0)], "y": []}
        rules = [
            _rule("x", 0, 1.0, 1.0, 0.0, 2.0),
            _rule("x", 0, 1.0, 1.25, 1.0, 2.0),
        ]

        result = validate_snap_plan(
            axes,
            rails,
            rules,
            raise_on_unsafe=False,
        )

        self.assertFalse(result.safe)
        self.assertEqual(result.failure.reason, "feature_not_represented")
        self.assertEqual(result.failure.rule_indices, (0,))

    def test_conflict_diagnostic_keeps_all_same_target_writers(self):
        axes = {"x": [0.0, 1.0, 2.0], "y": [0.0, 1.0]}
        rails = {"x": [_rail("x", 0, 1.0)], "y": []}
        rules = [
            _rule("x", 0, 1.0, 1.25, 0.0, 1.0),
            _rule("x", 0, 1.0, 1.25, 0.0, 1.0),
            _rule("x", 0, 1.0, 1.25, 0.0, 1.0),
            _rule("x", 0, 1.0, 1.5, 0.0, 1.0),
        ]

        result = validate_snap_plan(
            axes,
            rails,
            rules,
            raise_on_unsafe=False,
        )

        self.assertFalse(result.safe)
        self.assertEqual(result.failure.reason, "conflicting_node_targets")
        self.assertEqual(set(result.failure.rule_indices), {0, 1, 2, 3})

    def test_nested_full_span_noops_have_axis_independent_node_work(self):
        # This models an adversarial geometry-scale preflight: 1,000 nested z
        # lifetimes over 1,001 structural stations.  A no-op rule affects no
        # coordinates, so its span must remain a compact interval.  The work
        # counter is deterministic and avoids a machine-dependent time limit.
        axis = [float(index) for index in range(1001)]
        axes = {"x": axis, "y": axis}
        rails = {"x": [_rail("x", 0, 500.0)], "y": []}
        rules = [
            _rule(
                "x",
                0,
                500.0,
                500.0,
                0.0,
                1000.0,
                float(index),
                1000.0,
            )
            for index in range(1000)
        ]

        result = validate_snap_plan(axes, rails, rules)

        self.assertTrue(result.safe)
        self.assertEqual(result.critical_state_count, 2001)
        self.assertEqual(result.evaluated_state_count, 1)
        self.assertEqual(result.checked_cell_count, 0)
        self.assertEqual(result.enumerated_node_count, 0)

    def test_overlapping_moving_rules_do_not_store_rule_sized_node_sets(self):
        # All rules touch the same 1,001 nodes.  Validation must keep one
        # node-centric update map, not 1,000 persistent Python sets containing
        # the same full-span node IDs.  Peak-memory complexity is much less
        # machine-sensitive than a wall-clock assertion; the generous bound
        # leaves ample room for Python allocator/version differences.
        axis = [float(index) for index in range(1001)]
        axes = {"x": axis, "y": axis}
        rails = {"x": [_rail("x", 0, 500.0)], "y": []}
        rules = [
            _rule("x", 0, 500.0, 500.25, 0.0, 1000.0)
            for _ in range(1000)
        ]

        tracemalloc.start()
        try:
            result = validate_snap_plan(axes, rails, rules)
            _, peak_bytes = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        self.assertTrue(result.safe)
        self.assertEqual(result.enumerated_node_count, 1001)
        self.assertLess(peak_bytes, 16 * 1024 * 1024)

    def test_nested_identical_moving_lifecycles_validate_one_geometry(self):
        axis = [float(index) for index in range(1001)]
        axes = {"x": axis, "y": axis}
        rails = {"x": [_rail("x", 0, 500.0)], "y": []}
        rules = [
            _rule(
                "x",
                0,
                500.0,
                500.25,
                0.0,
                1000.0,
                float(index),
                1000.0,
            )
            for index in range(1000)
        ]

        result = validate_snap_plan(axes, rails, rules)

        self.assertTrue(result.safe)
        self.assertEqual(result.critical_state_count, 2001)
        self.assertEqual(result.evaluated_state_count, 1)
        self.assertEqual(result.enumerated_node_count, 1001)


if __name__ == "__main__":
    unittest.main()
