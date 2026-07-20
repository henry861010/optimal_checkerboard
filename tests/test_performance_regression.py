"""Complexity regressions for meshes that will be extruded to large 3D models.

These tests deliberately assert work-set sizes instead of elapsed time.  Wall
clock limits are noisy in CI, while accidentally returning to a full-mesh scan
is both deterministic and the scaling failure we need to prevent.
"""

import os
import sys
import unittest
import weakref
from unittest import mock

SRC_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "src")
)
sys.path.insert(0, SRC_ROOT)

import numpy as np

from optimal_checkerboard import Mesh2D, OptimalMesh25D
from optimal_checkerboard.algorithms import drag as drag_module
from optimal_checkerboard.algorithms.drag import Dragger
from optimal_checkerboard.mesh.checkerboard_mesh_box import (
    checkerboard_mesh_box,
)
from optimal_checkerboard.mesh.generator import (
    build_structured_rail_node_index,
)


class TestSparseSnapComplexity(unittest.TestCase):
    @staticmethod
    def _structured_mesher(axis_element_count=180):
        nodes, elements, x_nodes, y_nodes = checkerboard_mesh_box(
            [0.0, float(axis_element_count)],
            [0.0, float(axis_element_count)],
            element_size=1.0,
            return_axes=True,
        )
        nx = len(x_nodes)
        rail_coord = float(axis_element_count // 2)
        rail_column = int(np.searchsorted(x_nodes, rail_coord))
        rail_node_ids = rail_column + np.arange(len(y_nodes), dtype=np.intp) * nx

        mesher = OptimalMesh25D()
        mesher.mesh2d = Mesh2D(nodes=nodes, elements=elements)
        mesher.rails = {
            "x": [{"coord": rail_coord}],
            "y": [],
        }
        mesher.rail_node_index = {
            "x": [
                {
                    "coord": rail_coord,
                    "node_ids": rail_node_ids,
                    "span_values": y_nodes,
                }
            ],
            "y": [],
        }
        x_rail_ids = np.full(len(nodes), -1, dtype=np.intp)
        x_rail_ids[rail_node_ids] = 0
        mesher.node_axis_rail_ids = {
            "x": x_rail_ids,
            "y": np.full(len(nodes), -1, dtype=np.intp),
        }
        mesher._pattern_generation = 1
        mesher._configure_mesh_execution_state(structured=True)
        mesher.snap_rules_by_z = {
            0.0: [
                {
                    "axis": "x",
                    "rail_id": 0,
                    "z": 0.0,
                    "z_bottom": 0.0,
                    "z_top": 0.0,
                    "target_coord": rail_coord + 0.1,
                    "span_min": rail_coord,
                    "span_max": rail_coord + 1.0,
                }
            ]
        }
        mesher._pattern_integrity_signature = (
            mesher._current_pattern_integrity_signature()
        )
        mesher._record_mesh_integrity()
        return mesher, len(elements)

    def test_one_node_snap_validates_only_incident_elements(self):
        mesher, total_element_count = self._structured_mesher()
        validated_batch_sizes = []
        original_metrics = mesher._quad_validity_metrics

        def record_validation_work(nodes, element_ids):
            validated_batch_sizes.append(len(element_ids))
            return original_metrics(nodes, element_ids)

        with mock.patch.object(
            mesher,
            "_quad_validity_metrics",
            side_effect=record_validation_work,
        ):
            touched = mesher.apply_snap_rules_at_z(0.0)

        self.assertGreater(total_element_count, 30_000)
        self.assertEqual(touched, 2)
        self.assertEqual(validated_batch_sizes, [6])
        self.assertLess(validated_batch_sizes[0], total_element_count // 1_000)

    def test_snap_conflict_buffers_are_reused_between_z_events(self):
        mesher, _ = self._structured_mesher(axis_element_count=80)
        stamp_buffer = mesher._update_stamps
        target_buffer = mesher._update_targets

        mesher.apply_snap_rules_at_z(0.0)
        mesher.apply_snap_rules_at_z(1.0)

        self.assertIs(mesher._update_stamps, stamp_buffer)
        self.assertIs(mesher._update_targets, target_buffer)
        node_count = len(mesher.mesh2d.nodes)
        self.assertEqual(stamp_buffer.nbytes, node_count * 4)
        self.assertEqual(target_buffer.nbytes, node_count * 8)


class TestBoundedTemporaryMemory(unittest.TestCase):
    def test_many_disjoint_areas_use_one_spatial_prefilter(self):
        x_element_count = 120
        y_element_count = 80
        area_count = 40
        nodes, elements = checkerboard_mesh_box(
            [0.0, float(x_element_count)],
            [0.0, float(y_element_count)],
            element_size=1.0,
        )
        dragger = Dragger()
        dragger.set_2D(Mesh2D(nodes=nodes, elements=elements))
        stripe_width = x_element_count // area_count
        areas = [
            {
                "type": "BOX",
                "dim": [
                    float(area_index * stripe_width),
                    0.0,
                    float((area_index + 1) * stripe_width),
                    float(y_element_count),
                ],
                "material": "CORE",
            }
            for area_index in range(area_count)
        ]

        predicate_row_counts = []
        original_search = drag_module.search_face_element

        def record_predicate_rows(element_coordinates, *args, **kwargs):
            predicate_row_counts.append(len(element_coordinates))
            return original_search(element_coordinates, *args, **kwargs)

        with mock.patch.object(
            drag_module,
            "search_face_element",
            side_effect=record_predicate_rows,
        ), mock.patch.object(
            dragger,
            "_build_selector_spatial_index",
            wraps=dragger._build_selector_spatial_index,
        ) as build_index:
            dragger._organize(areas)

        element_count = len(elements)
        brute_force_rows = area_count * element_count
        exact_predicate_rows = sum(predicate_row_counts)
        self.assertEqual(
            len(dragger._element_2D_active_indices),
            element_count,
        )
        # All exact inclusion checks remain in place, but their aggregate rows
        # stay near E plus boundary-bin spill instead of areas * E.
        self.assertLess(exact_predicate_rows, 3 * element_count)
        self.assertLess(exact_predicate_rows, brute_force_rows // 10)
        self.assertEqual(build_index.call_count, 1)
        self.assertIsNone(dragger._selector_spatial_index)

    def test_holes_contribute_to_spatial_index_threshold_only(self):
        holes = [
            {
                "type": "BOX",
                "dim": [float(index), 0.0, float(index + 1), 1.0],
            }
            for index in range(8)
        ]
        areas = [
            {
                "type": "CYLINDER",
                "dim": [4.0, 0.5, 5.0],
                "material": "CORE",
                "holes": holes[:5],
                "metals": [
                    {
                        "type": "NORMAL",
                        "material": "M",
                        "density": 0.0,
                        "holes": holes[5:],
                    }
                ],
            }
        ]

        self.assertEqual(Dragger._indexable_selector_count(areas), 8)

    def test_spatial_prefilter_matches_brute_force_for_randomized_selectors(self):
        rng = np.random.default_rng(20260721)
        nodes, elements = checkerboard_mesh_box(
            [0.0, 60.0],
            [0.0, 45.0],
            element_size=1.0,
        )
        # Exercise current-node anchor indexing with nonuniform,
        # non-axis-aligned elements while keeping all coordinates finite.
        nodes = nodes.copy()
        interior = (
            (nodes[:, 0] > 0.0)
            & (nodes[:, 0] < 60.0)
            & (nodes[:, 1] > 0.0)
            & (nodes[:, 1] < 45.0)
        )
        nodes[interior, :2] += rng.uniform(
            -0.12,
            0.12,
            size=(int(interior.sum()), 2),
        )
        dragger = Dragger()
        dragger.set_2D(Mesh2D(nodes=nodes, elements=elements))
        spatial_index = dragger._build_selector_spatial_index(
            selector_count=24
        )
        self.assertIsNotNone(spatial_index)

        polygon_with_hole = {
            "type": "POLYGON",
            "dim": [
                [
                    [8.0, 6.0],
                    [8.0, 39.0],
                    [52.0, 39.0],
                    [52.0, 6.0],
                ],
                [
                    [22.0, 17.0],
                    [37.0, 17.0],
                    [37.0, 28.0],
                    [22.0, 28.0],
                ],
            ],
        }

        try:
            for iteration in range(20):
                first_x = float(rng.uniform(-5.0, 40.0))
                first_y = float(rng.uniform(-5.0, 30.0))
                ranges = [
                    {
                        "type": "BOX",
                        "dim": [
                            first_x,
                            first_y,
                            first_x + float(rng.uniform(2.0, 20.0)),
                            first_y + float(rng.uniform(2.0, 16.0)),
                        ],
                    },
                    polygon_with_hole,
                ]
                hole_x = float(rng.uniform(4.0, 48.0))
                hole_y = float(rng.uniform(4.0, 34.0))
                holes = [
                    {
                        "type": "BOX",
                        "dim": [
                            hole_x,
                            hole_y,
                            hole_x + float(rng.uniform(1.0, 7.0)),
                            hole_y + float(rng.uniform(1.0, 7.0)),
                        ],
                    }
                ]
                if iteration % 2:
                    element_indices = np.sort(
                        rng.choice(
                            len(elements),
                            size=len(elements) * 2 // 3,
                            replace=False,
                        )
                    ).astype(np.int32)
                else:
                    element_indices = None

                dragger._selector_spatial_index = None
                brute_hits = dragger._search_faces(
                    element_indices,
                    ranges,
                    holes,
                    chunk_size=127,
                )
                brute_mask = dragger._search_faces(
                    element_indices,
                    ranges,
                    holes,
                    returnMask=True,
                    chunk_size=127,
                )

                dragger._selector_spatial_index = spatial_index
                indexed_hits = dragger._search_faces(
                    element_indices,
                    ranges,
                    holes,
                    chunk_size=127,
                )
                indexed_mask = dragger._search_faces(
                    element_indices,
                    ranges,
                    holes,
                    returnMask=True,
                    chunk_size=127,
                )

                np.testing.assert_array_equal(indexed_hits, brute_hits)
                np.testing.assert_array_equal(indexed_mask, brute_mask)
                if len(indexed_hits) > 1:
                    self.assertTrue(np.all(indexed_hits[1:] > indexed_hits[:-1]))

            # An arbitrary subset cannot be mapped through searchsorted while
            # preserving local semantics, so it must take the full-scan path.
            sorted_subset = np.sort(
                rng.choice(len(elements), size=700, replace=False)
            ).astype(np.int32)
            unsorted_subset = sorted_subset[::-1].copy()
            self.assertIsNone(
                spatial_index.candidate_local_indices(
                    [polygon_with_hole],
                    unsorted_subset,
                )
            )
            dragger._selector_spatial_index = None
            brute_unsorted = dragger._search_faces(
                unsorted_subset,
                [polygon_with_hole],
            )
            dragger._selector_spatial_index = spatial_index
            indexed_unsorted = dragger._search_faces(
                unsorted_subset,
                [polygon_with_hole],
            )
            np.testing.assert_array_equal(indexed_unsorted, brute_unsorted)
        finally:
            dragger._selector_spatial_index = None

    def test_spatial_anchor_keeps_extreme_one_ulp_boundary_hits(self):
        base_x = np.float64(1.0e150)
        base_y = np.float64(-1.0e150)
        ulp_x = np.spacing(base_x)
        ulp_y = abs(np.spacing(base_y))
        second_x = base_x + 32.0 * ulp_x
        nodes = np.array(
            [
                [base_x, base_y],
                [base_x + ulp_x, base_y],
                [base_x + ulp_x, base_y + ulp_y],
                [base_x, base_y + ulp_y],
                [second_x, base_y],
                [second_x + ulp_x, base_y],
                [second_x + ulp_x, base_y + ulp_y],
                [second_x, base_y + ulp_y],
            ],
            dtype=np.float64,
        )
        elements = np.array(
            [[0, 1, 2, 3], [4, 5, 6, 7]],
            dtype=np.int32,
        )
        dragger = Dragger()
        dragger.set_2D(Mesh2D(nodes=nodes, elements=elements))
        spatial_index = dragger._build_selector_spatial_index(
            selector_count=8
        )
        self.assertIsNotNone(spatial_index)
        selector = [
            {
                "type": "BOX",
                "dim": [
                    base_x,
                    base_y,
                    base_x + ulp_x,
                    base_y + ulp_y,
                ],
            }
        ]

        dragger._selector_spatial_index = None
        brute_hits = dragger._search_faces(ranges=selector)
        dragger._selector_spatial_index = spatial_index
        indexed_hits = dragger._search_faces(ranges=selector)
        dragger._selector_spatial_index = None

        np.testing.assert_array_equal(brute_hits, [0])
        np.testing.assert_array_equal(indexed_hits, brute_hits)

    def test_spatial_index_extreme_aspect_ratio_safely_falls_back(self):
        element_count = 4_096
        nodes, elements = checkerboard_mesh_box(
            [0.0, 1.0],
            [0.0, 1.0e-306],
            element_size=1.0 / element_count,
        )
        elements = elements.copy()
        # A cyclic start-point change preserves each quad while making anchor
        # y span the two physical rails.  Only the bin-layout heuristic, not
        # the valid geometry, has an unrepresentable aspect calculation.
        elements[1::2] = np.roll(elements[1::2], 2, axis=1)
        dragger = Dragger()
        dragger.set_2D(Mesh2D(nodes=nodes, elements=elements))

        spatial_index = dragger._build_selector_spatial_index(selector_count=8)

        self.assertIsNone(spatial_index)
        hits = dragger._search_faces(
            ranges=[{
                "type": "BOX",
                "dim": [0.0, 0.0, 1.0, 1.0e-306],
            }]
        )
        np.testing.assert_array_equal(
            hits,
            np.arange(element_count, dtype=np.int32),
        )

    def test_polygon_boundary_broadcast_obeys_matrix_entry_budget(self):
        angles = np.linspace(0.0, 2.0 * np.pi, 101, endpoint=False)
        vertices = np.column_stack((np.cos(angles), np.sin(angles)))
        points = np.column_stack(
            (
                np.linspace(-2.0, 2.0, 25),
                np.full(25, 0.123, dtype=np.float64),
            )
        )
        point_batch_sizes = []
        original_chunk = drag_module._point_chunk_on_loop_boundary

        def record_boundary_chunk(point_chunk, *args, **kwargs):
            point_batch_sizes.append(len(point_chunk))
            return original_chunk(point_chunk, *args, **kwargs)

        matrix_entry_budget = 1_000
        with mock.patch.object(
            drag_module,
            "_POLYGON_BOUNDARY_MATRIX_ENTRY_BUDGET",
            matrix_entry_budget,
        ), mock.patch.object(
            drag_module,
            "_point_chunk_on_loop_boundary",
            side_effect=record_boundary_chunk,
        ):
            result = drag_module._points_on_loop_boundary(
                points,
                vertices,
                chunk_size=len(points),
            )

        self.assertEqual(result.shape, (len(points),))
        self.assertGreater(len(point_batch_sizes), 1)
        self.assertLessEqual(
            max(point_batch_sizes) * len(vertices),
            matrix_entry_budget,
        )

    def test_face_search_obeys_requested_chunk_size(self):
        element_count = 2_003
        nodes, elements = checkerboard_mesh_box(
            [0.0, float(element_count)],
            [0.0, 1.0],
            element_size=1.0,
        )
        dragger = Dragger()
        dragger.set_2D(Mesh2D(nodes=nodes, elements=elements))
        predicate_batch_sizes = []
        original_search = drag_module.search_face_element

        def record_search(element_coordinates, *args, **kwargs):
            predicate_batch_sizes.append(len(element_coordinates))
            return original_search(element_coordinates, *args, **kwargs)

        with mock.patch.object(
            drag_module,
            "search_face_element",
            side_effect=record_search,
        ):
            hits = dragger._search_faces(
                ranges=[{
                    "type": "BOX",
                    "dim": [0.0, 0.0, float(element_count), 1.0],
                }],
                chunk_size=113,
            )

        self.assertTrue(predicate_batch_sizes)
        self.assertLessEqual(max(predicate_batch_sizes), 113)
        np.testing.assert_array_equal(
            hits,
            np.arange(element_count, dtype=np.int32),
        )

    def test_quad_validation_obeys_requested_chunk_size(self):
        element_count = 2_003
        nodes, elements = checkerboard_mesh_box(
            [0.0, float(element_count)],
            [0.0, 1.0],
            element_size=1.0,
        )
        dragger = Dragger()
        dragger.node_2D = nodes[:, :2]
        dragger.element_2D = elements
        dragger.element_2D_volumn = np.empty(element_count, dtype=np.float64)

        cross_batch_sizes = []
        original_cross = drag_module._cross_2d

        def record_cross(vector_a, vector_b):
            cross_batch_sizes.append(len(vector_a))
            return original_cross(vector_a, vector_b)

        with mock.patch.object(
            drag_module,
            "_cross_2d",
            side_effect=record_cross,
        ):
            dragger._cal_volumns(chunk_size=113)

        self.assertTrue(cross_batch_sizes)
        self.assertLessEqual(max(cross_batch_sizes), 113)
        np.testing.assert_allclose(dragger.element_2D_volumn, 1.0)

    def test_structured_rail_index_materializes_only_rail_length(self):
        # A 500,001 x 5 structured grid represents 2.5M nodes.  One vertical
        # rail needs five node ids, not a full node or connectivity scan.
        x_nodes = np.arange(500_001, dtype=np.float64)
        y_nodes = np.arange(5, dtype=np.float64)
        index = build_structured_rail_node_index(
            x_nodes,
            y_nodes,
            {"x": [{"coord": 250_000.0}], "y": []},
        )

        rail = index["x"][0]
        self.assertEqual(len(rail["node_ids"]), len(y_nodes))
        self.assertEqual(rail["node_ids"].nbytes, len(y_nodes) * 8)
        np.testing.assert_allclose(rail["span_values"], y_nodes)

    def test_drag_connectivity_chunks_flat_final_hex_indices(self):
        nodes, elements = checkerboard_mesh_box(
            [0.0, 2.0],
            [0.0, 1.0],
            element_size=1.0,
        )
        dragger = Dragger()
        dragger.set_2D(Mesh2D(nodes=nodes, elements=elements))
        dragger._organize({
            "type": "BOX",
            "dim": [0.0, 0.0, 2.0, 1.0],
            "material": "CORE",
        })

        chunk_planes = []
        original_writer = drag_module._write_drag_connectivity_chunk

        def record_connectivity_chunk(
            destination,
            local_connectivity,
            plane_indices,
            *args,
            **kwargs,
        ):
            chunk_planes.append(plane_indices.copy())
            return original_writer(
                destination,
                local_connectivity,
                plane_indices,
                *args,
                **kwargs,
            )

        with mock.patch.object(
            drag_module,
            "_CONNECTIVITY_CHUNK_SIZE",
            3,
        ), mock.patch.object(
            drag_module,
            "_write_drag_connectivity_chunk",
            side_effect=record_connectivity_chunk,
        ):
            drag_count = dragger._drag(0.25, 0.0, 1.0)

        self.assertEqual(drag_count, 4)
        self.assertEqual(dragger.element_num, 8)
        self.assertEqual(len(chunk_planes), 3)
        self.assertLessEqual(max(map(len, chunk_planes)), 3)
        np.testing.assert_array_equal(
            np.concatenate(chunk_planes),
            [0, 0, 1, 1, 2, 2, 3, 3],
        )
        for flat_hex_id, connectivity in enumerate(
            dragger.elements[:dragger.element_num]
        ):
            plane = flat_hex_id // len(elements)
            np.testing.assert_allclose(
                dragger.nodes[connectivity[:4], 2],
                plane * 0.25,
            )
            np.testing.assert_allclose(
                dragger.nodes[connectivity[4:], 2],
                (plane + 1) * 0.25,
            )

    def test_drag_uses_active_elements_and_sparse_node_mapping_reset(self):
        element_count = 2_003
        nodes, elements = checkerboard_mesh_box(
            [0.0, float(element_count)],
            [0.0, 1.0],
            element_size=1.0,
        )
        dragger = Dragger()
        dragger.set_2D(Mesh2D(nodes=nodes, elements=elements))
        area = {
            "type": "BOX",
            "dim": [0.0, 0.0, 1.0, 1.0],
            "material": "CORE",
        }
        cleared_mapping_sizes = []
        original_clear = dragger._clear_tracked_node_mapping

        def record_sparse_clear():
            cleared_mapping_sizes.append(
                len(dragger._mapped_node_2D_indices)
            )
            return original_clear()

        dragger._organize(area)
        with mock.patch.object(
            drag_module.np,
            "flatnonzero",
            side_effect=AssertionError("_drag scanned the full component array"),
        ), mock.patch.object(
            dragger,
            "_clear_tracked_node_mapping",
            side_effect=record_sparse_clear,
        ):
            dragger._drag(1.0, 0.0, 1.0)

        dragger._organize(area)
        with mock.patch.object(
            drag_module.np,
            "flatnonzero",
            side_effect=AssertionError("_drag scanned the full component array"),
        ), mock.patch.object(
            dragger,
            "_clear_tracked_node_mapping",
            side_effect=record_sparse_clear,
        ):
            dragger._drag(1.0, 1.0, 2.0)

        self.assertGreater(len(nodes), 4_000)
        self.assertEqual(dragger.element_num, 2)
        self.assertEqual(cleared_mapping_sizes, [0, 4])
        self.assertEqual(len(dragger._mapped_node_2D_indices), 4)

    def test_normal_metal_selector_is_evaluated_once_and_removal_is_mask_based(self):
        element_count = 20
        nodes, elements = checkerboard_mesh_box(
            [0.0, float(element_count)],
            [0.0, 1.0],
            element_size=1.0,
        )
        dragger = Dragger()
        dragger.set_2D(Mesh2D(nodes=nodes, elements=elements))
        area = {
            "type": "BOX",
            "dim": [0.0, 0.0, float(element_count), 1.0],
            "material": "BASE",
            "metals": [
                {
                    "type": "NORMAL",
                    "material": "M1",
                    "density": 50.0,
                    "ranges": [{"type": "BOX", "dim": [0.0, 0.0, 10.0, 1.0]}],
                },
                {
                    "type": "NORMAL",
                    "material": "M2",
                    "density": 50.0,
                    "ranges": [{"type": "BOX", "dim": [10.0, 0.0, 20.0, 1.0]}],
                },
            ],
        }
        original_search = dragger._search_faces

        with mock.patch.object(
            dragger,
            "_search_faces",
            wraps=original_search,
        ) as search_mock, mock.patch.object(
            drag_module.np,
            "setdiff1d",
            side_effect=AssertionError("remaining pool used setdiff1d"),
        ), mock.patch.object(
            drag_module.np,
            "isin",
            side_effect=AssertionError("remaining pool used isin"),
        ):
            dragger._organize(area, layer=7)

        # One area selector plus one selector per metal.  NORMAL selectors are
        # not scanned again during assignment.
        self.assertEqual(search_mock.call_count, 3)
        counts = np.bincount(
            dragger.element_2D_comp,
            minlength=len(dragger.comps),
        )
        self.assertEqual(counts[dragger.comps["M1"]], 5)
        self.assertEqual(counts[dragger.comps["M2"]], 5)
        self.assertEqual(counts[dragger.comps["BASE"]], 10)

    def test_metal_selector_hits_are_released_between_priority_items(self):
        nodes, elements = checkerboard_mesh_box(
            [0.0, 5.0],
            [0.0, 1.0],
            element_size=1.0,
        )
        dragger = Dragger()
        dragger.set_2D(Mesh2D(nodes=nodes, elements=elements))
        selector = [{"type": "BOX", "dim": [0.0, 0.0, 5.0, 1.0]}]
        area = {
            "type": "BOX",
            "dim": [0.0, 0.0, 5.0, 1.0],
            "material": "BASE",
            "metals": [
                {
                    "type": "NORMAL",
                    "material": f"M{index}",
                    "density": 0.0,
                    "ranges": selector,
                }
                for index in range(32)
            ],
        }
        original_search = dragger._search_faces
        selector_refs = []
        live_before_search = []
        call_count = 0

        def track_selector_lifetime(*args, **kwargs):
            nonlocal call_count
            # The first search selects the area and must remain live.  Track
            # only per-metal selector arrays.
            if call_count:
                live_before_search.append(
                    sum(ref() is not None for ref in selector_refs)
                )
            result = original_search(*args, **kwargs)
            if call_count:
                selector_refs.append(weakref.ref(result))
            call_count += 1
            return result

        with mock.patch.object(
            dragger,
            "_search_faces",
            side_effect=track_selector_lifetime,
        ):
            dragger._organize(area)

        self.assertEqual(call_count, 33)
        # At the next allocation only the immediately previous selector may
        # still be bound on the assignment RHS.  Hit arrays never accumulate
        # with the number of geometry selectors.
        self.assertLessEqual(max(live_before_search), 1)


if __name__ == "__main__":
    unittest.main()
