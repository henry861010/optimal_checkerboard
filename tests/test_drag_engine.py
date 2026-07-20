import os
import sys
import unittest

SRC_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "src")
)
sys.path.insert(0, SRC_ROOT)

import numpy as np

from optimal_checkerboard.algorithms.drag import Engin25D, search_face_element


class SimpleMesh2D:
    def __init__(self):
        self.nodes = np.asarray(
            [
                [0, 0, 0],
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 0],
                [2, 0, 0],
                [2, 1, 0],
            ],
            dtype=np.float32,
        )
        self.elements = np.asarray(
            [
                [0, 1, 2, 3],
                [1, 4, 5, 2],
            ],
            dtype=np.int32,
        )


class SingleQuadMesh2D:
    def __init__(self):
        self.nodes = np.asarray(
            [
                [-1, -1, 0],
                [1, -1, 0],
                [1, 1, 0],
                [-1, 1, 0],
            ],
            dtype=np.float64,
        )
        self.elements = np.asarray([[0, 1, 2, 3]], dtype=np.int32)


class TestDragEngine(unittest.TestCase):
    def test_search_polygon_uses_winding_hulls_holes_and_inclusive_boundary(self):
        elements = np.asarray(
            [
                [0, 0, 1, 0, 1, 1, 0, 1],
                [2, 2, 3, 2, 3, 3, 2, 3],
                [10, 0, 11, 0, 11, 1, 10, 1],
                [20, 0, 21, 0, 21, 1, 20, 1],
            ],
            dtype=np.float64,
        )
        dim = [
            [[0, 0], [0, 5], [5, 5], [5, 0]],
            [[2, 2], [3, 2], [3, 3], [2, 3]],
            [[10, 0], [10, 5], [15, 5], [15, 0]],
        ]

        hits = search_face_element(elements, "POLYGON", dim)

        np.testing.assert_array_equal(hits, [0, 2])

    def test_search_polygon_keeps_elements_outside_holes_on_the_boundary(self):
        elements = np.asarray(
            [
                [1, 2, 2, 2, 2, 3, 1, 3],
                [1, 1, 2, 1, 2, 2, 1, 2],
                [2, 2, 3, 2, 3, 3, 2, 3],
                [5.2, 2.2, 5.8, 2.2, 5.8, 2.8, 5.2, 2.8],
                [3, 2, 5, 2, 5, 3, 3, 3],
                [1.5, 1.5, 2.5, 1.5, 2.5, 2.5, 1.5, 2.5],
            ],
            dtype=np.float64,
        )
        dim = [
            [[0, 0], [0, 6], [8, 6], [8, 0]],
            [[2, 2], [3, 2], [3, 3], [2, 3]],
            [[5, 2], [6, 2], [6, 3], [5, 3]],
        ]

        hits = search_face_element(elements, "POLYGON", dim)

        np.testing.assert_array_equal(hits, [0, 1, 4])

    def test_search_polygon_does_not_combine_corners_from_disjoint_hulls(self):
        elements = np.asarray(
            [[0.5, 0.2, 3.5, 0.2, 3.5, 0.8, 0.5, 0.8]],
            dtype=np.float64,
        )
        dim = [
            [[0, 0], [0, 1], [1, 1], [1, 0]],
            [[3, 0], [3, 1], [4, 1], [4, 0]],
        ]

        hits = search_face_element(elements, "POLYGON", dim)

        self.assertEqual(len(hits), 0)

    def test_polygon_boundary_does_not_swallow_thin_large_coordinate_cell(self):
        """Scale-derived eps must not leak material over an exact boundary."""
        boundary_x = 1_000_000.0
        outside_x = boundary_x + 5e-7
        elements = np.asarray(
            [
                [999_999.0, 0.0, boundary_x, 0.0,
                 boundary_x, 1.0, 999_999.0, 1.0],
                [boundary_x, 0.0, outside_x, 0.0,
                 outside_x, 1.0, boundary_x, 1.0],
            ],
            dtype=np.float64,
        )
        polygon = [[
            [999_999.0, 0.0],
            [999_999.0, 1.0],
            [boundary_x, 1.0],
            [boundary_x, 0.0],
        ]]

        hits = search_face_element(elements, "POLYGON", polygon)

        np.testing.assert_array_equal(hits, [0])

    def test_cylinder_search_is_safe_when_squared_values_would_overflow(self):
        radius = 1e200
        elements = np.asarray(
            [[
                2.0 * radius, 0.0,
                2.1 * radius, 0.0,
                2.1 * radius, 0.1 * radius,
                2.0 * radius, 0.1 * radius,
            ]],
            dtype=np.float64,
        )

        hits = search_face_element(
            elements,
            "CYLINDER",
            [0.0, 0.0, radius],
        )

        self.assertEqual(len(hits), 0)

    def test_assign_metal_takes_minimum_random_prefix_to_reach_target(self):
        engine = Engin25D()
        volumes = np.ones(4, dtype=np.float64)
        expected_order = np.random.default_rng(7).permutation(len(volumes))

        chosen = engine._assign_metal(
            volumes,
            density=50,
            total_volume=4,
            randomSeed=7,
        )

        np.testing.assert_array_equal(chosen, expected_order[:2])

        chosen = engine._assign_metal(
            volumes,
            density=50,
            total_volume=1,
            randomSeed=7,
        )
        np.testing.assert_array_equal(chosen, expected_order[:1])

    def test_assign_metal_handles_nonpositive_and_unreachable_targets(self):
        engine = Engin25D()
        volumes = np.asarray([1.0, 2.0, 3.0])

        for density in (0, -1):
            with self.subTest(density=density):
                chosen = engine._assign_metal(
                    volumes,
                    density=density,
                    total_volume=volumes.sum(),
                    randomSeed=3,
                )
                self.assertEqual(chosen.dtype, np.int32)
                self.assertEqual(len(chosen), 0)

        expected_order = np.random.default_rng(3).permutation(len(volumes))
        chosen = engine._assign_metal(
            volumes,
            density=200,
            total_volume=volumes.sum(),
            randomSeed=3,
        )
        np.testing.assert_array_equal(chosen, expected_order)

    def test_set_2d_uses_split_element_arrays(self):
        mesh = SimpleMesh2D()
        engine = Engin25D()

        engine.set_2D(mesh)

        self.assertFalse(hasattr(engine, "element_internal"))
        np.testing.assert_array_equal(engine.element_2D, mesh.elements)
        np.testing.assert_allclose(engine.element_2D_volumn, [1.0, 1.0])
        np.testing.assert_array_equal(engine.element_2D_comp, [0, 0])

    def test_set_2d_rejects_lossy_connectivity_conversion(self):
        for elements in (
            np.asarray([[0.0, 1.0, 2.0, 3.5]], dtype=np.float64),
            np.asarray([[2**32, 1, 2, 3]], dtype=np.uint64),
        ):
            with self.subTest(dtype=elements.dtype, value=elements[0, 0]):
                mesh = SingleQuadMesh2D()
                mesh.elements = elements
                with self.assertRaises((ValueError, OverflowError)):
                    Engin25D().set_2D(mesh)

    def test_organize_and_drag_use_split_element_arrays(self):
        engine = Engin25D()
        engine.set_2D(SimpleMesh2D())

        local_hits = engine._search_faces(
            np.asarray([1], dtype=np.int32),
            ranges=[{"type": "BOX", "dim": [1, 0, 2, 1]}],
        )
        np.testing.assert_array_equal(local_hits, [0])

        engine._organize(
            {
                "type": "BOX",
                "dim": [0, 0, 1, 1],
                "material": "CORE",
            }
        )

        core_id = engine.comps["CORE"]
        np.testing.assert_array_equal(engine.element_2D_comp, [core_id, 0])

        engine._drag(element_size=0.5, begin=0.0, end=1.0)

        self.assertEqual(engine.element_num, 2)
        self.assertEqual(engine.node_num, 12)
        np.testing.assert_array_equal(
            engine.element_comps[:engine.element_num],
            [core_id, core_id],
        )

    def test_smaller_layer_footprint_does_not_extrude_previous_material(self):
        engine = Engin25D()
        engine.set_2D(SimpleMesh2D())
        full_area = {
            "type": "BOX",
            "dim": [0, 0, 2, 1],
            "material": "CORE",
        }
        left_area = {
            "type": "BOX",
            "dim": [0, 0, 1, 1],
            "material": "CORE",
        }

        engine._organize(full_area)
        engine._drag(element_size=1, begin=0, end=1)
        core_id = engine.comps["CORE"]
        np.testing.assert_array_equal(engine.element_2D_comp, [core_id, core_id])

        engine._organize(left_area)

        np.testing.assert_array_equal(
            engine.previous_element_2D_comp,
            [core_id, core_id],
        )
        np.testing.assert_array_equal(engine.element_2D_comp, [core_id, 0])
        engine._drag(element_size=1, begin=1, end=2)

        self.assertEqual(engine.element_num, 3)
        last_hex = engine.elements[engine.element_num - 1]
        last_hex_xy = engine.nodes[last_hex, :2]
        self.assertLessEqual(float(last_hex_xy[:, 0].max()), 1.0)

    def test_organize_rejects_overlapping_area_ownership(self):
        engine = Engin25D()
        engine.set_2D(SimpleMesh2D())
        area = {
            "type": "BOX",
            "dim": [0.0, 0.0, 2.0, 1.0],
            "material": "CORE",
        }

        with self.assertRaisesRegex(ValueError, "areas overlap"):
            engine._organize([area, dict(area)])

    def test_continue_and_convert_read_previous_layer_snapshot(self):
        engine = Engin25D()
        engine.set_2D(SimpleMesh2D())
        left = {"type": "BOX", "dim": [0, 0, 1, 1], "material": "M1"}
        right = {"type": "BOX", "dim": [1, 0, 2, 1], "material": "BASE"}
        engine._organize([left, right])

        engine._organize(
            {
                "type": "BOX",
                "dim": [0, 0, 2, 1],
                "material": "CORE",
                "metals": [
                    {
                        "type": "CONTINUE",
                        "material": "M1",
                        "ranges": [{"type": "BOX", "dim": [0, 0, 1, 1]}],
                    },
                    {
                        "type": "CONVERT",
                        "material_o": "BASE",
                        "material": "M2",
                        "ranges": [{"type": "BOX", "dim": [1, 0, 2, 1]}],
                    },
                ],
            }
        )

        np.testing.assert_array_equal(
            engine.element_2D_comp,
            [engine.comps["M1"], engine.comps["M2"]],
        )

    def test_drag_uses_ceil_subdivision_and_preserves_exact_end_plane(self):
        engine = Engin25D()
        engine.set_2D(SimpleMesh2D())
        engine._organize(
            {"type": "BOX", "dim": [0, 0, 1, 1], "material": "CORE"}
        )

        drag_count = engine._drag(element_size=0.6, begin=0.0, end=1.0)

        self.assertEqual(drag_count, 2)
        self.assertEqual(engine.element_num, 2)
        z_values = np.unique(engine.nodes[:engine.node_num, 2])
        np.testing.assert_allclose(z_values, [0.0, 0.5, 1.0])
        self.assertEqual(float(z_values[-1]), 1.0)

    def test_drag_keeps_positive_sub_roundoff_layer(self):
        engine = Engin25D()
        engine.set_2D(SimpleMesh2D())
        engine._organize(
            {"type": "BOX", "dim": [0, 0, 1, 1], "material": "CORE"}
        )
        end = 1e-7

        drag_count = engine._drag(element_size=1.0, begin=0.0, end=end)

        self.assertEqual(drag_count, 1)
        self.assertEqual(float(engine.nodes[:engine.node_num, 2].max()), end)

    def test_drag_rejects_invalid_interval_and_element_size(self):
        engine = Engin25D()
        for element_size, begin, end, message in (
            (1.0, 1.0, 1.0, "begin < end"),
            (1.0, 2.0, 1.0, "begin < end"),
            (0.0, 0.0, 1.0, "element_size"),
            (-1.0, 0.0, 1.0, "element_size"),
            (float("nan"), 0.0, 1.0, "element_size"),
            (1.0, float("nan"), 1.0, "finite"),
        ):
            with self.subTest(element_size=element_size, begin=begin, end=end):
                with self.assertRaisesRegex(ValueError, message):
                    engine._drag(element_size, begin, end)

    def test_drag_rejects_unrepresentable_duplicate_z_planes(self):
        engine = Engin25D()
        engine.set_2D(SingleQuadMesh2D())
        engine._organize(
            {"type": "BOX", "dim": [-1, -1, 1, 1], "material": "CORE"}
        )
        begin = 1e16

        with self.assertRaisesRegex(ValueError, "float64 z resolution"):
            engine._drag(
                element_size=1.0,
                begin=begin,
                end=begin + 4.0,
            )
        self.assertEqual(engine.node_num, 0)
        self.assertEqual(engine.element_num, 0)

    def test_drag_accepts_normal_decimal_subdivision_without_zero_planes(self):
        engine = Engin25D()
        engine.set_2D(SingleQuadMesh2D())
        engine._organize(
            {"type": "BOX", "dim": [-1, -1, 1, 1], "material": "CORE"}
        )

        count = engine._drag(element_size=0.1, begin=0.0, end=1.0)
        z_values = np.unique(engine.nodes[:engine.node_num, 2])

        self.assertEqual(count, 10)
        self.assertTrue(np.all(np.diff(z_values) > 0.0))

    def test_set_2d_rejects_clockwise_and_degenerate_quads(self):
        for elements, message in (
            ([[0, 3, 2, 1]], "non-positive signed area"),
            ([[0, 1, 1, 3]], "degenerate"),
        ):
            mesh = SimpleMesh2D()
            mesh.elements = np.asarray(elements, dtype=np.int32)
            with self.subTest(elements=elements):
                with self.assertRaisesRegex(ValueError, message):
                    Engin25D().set_2D(mesh)

    def test_top_plane_update_rejects_interior_transition_collapse(self):
        engine = Engin25D()
        engine.set_2D(SingleQuadMesh2D())
        engine._organize(
            {"type": "BOX", "dim": [-1, -1, 1, 1], "material": "CORE"}
        )
        engine._drag(element_size=1, begin=0, end=1)
        top_node_ids = engine.node_2D_to_3D.copy()
        before = engine.nodes[top_node_ids, :2].copy()

        # Both endpoint quads are the same strictly CCW square.  The labels on
        # the prospective top are rotated by 180 degrees, so linear extrusion
        # collapses all four corners at t=0.5.
        prospective_xy = engine.node_2D[[2, 3, 0, 1], :2]
        engine.node_2D[:] = prospective_xy
        engine._cal_volumns()  # Endpoint alone is a valid CCW convex quad.

        with self.assertRaisesRegex(ValueError, "transition"):
            engine.validate_top_plane_xy_update(
                np.arange(4),
                prospective_xy,
                chunk_size=1,
            )
        with self.assertRaisesRegex(ValueError, "transition"):
            engine.sync_top_plane_xy(
                np.arange(4),
                prospective_xy,
                incident_element_ids=[0],
                chunk_size=1,
            )

        np.testing.assert_allclose(engine.nodes[top_node_ids, :2], before)

    def test_top_plane_update_rejects_interior_transition_inversion(self):
        engine = Engin25D()
        engine.set_2D(SingleQuadMesh2D())
        engine._organize(
            {"type": "BOX", "dim": [-1, -1, 1, 1], "material": "CORE"}
        )
        engine._drag(element_size=1, begin=0, end=1)
        prospective_xy = np.asarray(
            [[2, 2], [1, 2], [-3, 1], [1, 0]],
            dtype=np.float64,
        )
        engine.node_2D[:] = prospective_xy
        engine._cal_volumns()  # Both z endpoints remain strictly convex/CCW.

        with self.assertRaisesRegex(ValueError, "inverted transition"):
            engine.sync_top_plane_xy(np.arange(4), prospective_xy)

    def test_top_plane_update_commits_only_after_safe_transition(self):
        engine = Engin25D()
        engine.set_2D(SingleQuadMesh2D())
        engine._organize(
            {"type": "BOX", "dim": [-1, -1, 1, 1], "material": "CORE"}
        )
        engine._drag(element_size=1, begin=0, end=1)
        prospective_xy = engine.node_2D + np.asarray([0.5, 0.25])

        validated = engine.validate_top_plane_xy_update(
            np.arange(4),
            prospective_xy,
        )
        updated = engine.sync_top_plane_xy(
            np.arange(4),
            prospective_xy,
        )

        self.assertEqual(validated, 1)
        self.assertEqual(updated, 4)
        np.testing.assert_allclose(
            engine.nodes[engine.node_2D_to_3D, :2],
            prospective_xy,
        )


if __name__ == "__main__":
    unittest.main()
