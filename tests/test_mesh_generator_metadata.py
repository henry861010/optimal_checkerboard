import importlib
import os
import sys
import unittest
from unittest import mock

SRC_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "src")
)
sys.path.insert(0, SRC_ROOT)

import numpy as np

mesh_box_module = importlib.import_module(
    "optimal_checkerboard.mesh.checkerboard_mesh_box"
)
from optimal_checkerboard.algorithms.feature_lines import (
    feature_span_endpoint_coordinates,
    line_span_endpoint_coordinates,
)
from optimal_checkerboard.mesh.checkerboard_mesh_box import (
    _mesh_elements,
    _mesh_nodes,
    checkerboard_mesh_box,
)
from optimal_checkerboard.mesh.domain import (
    mesh_domain_pinned_coordinates,
    normalize_mesh_domain,
)
from optimal_checkerboard.mesh.generator import (
    build_structured_rail_node_index,
    generate_checkerboard_mesh_box,
)


class TestMeshGeneratorMetadata(unittest.TestCase):
    def test_finite_line_endpoints_become_mandatory_cross_axis_nodes(self):
        faces = [
            {
                "type": "LINE",
                "dim": [4, 2, 4, 8],
                "bottom_z": 0,
                "top_z": 5,
            },
            {
                "type": "LINE",
                "dim": [5, 2, 5, 8],
                "bottom_z": 10,
                "top_z": 15,
            },
        ]
        mandatory = line_span_endpoint_coordinates(faces)
        domain = normalize_mesh_domain(
            {"type": "BOX", "dim": [0, 0, 10, 10]}
        )

        nodes, elements, mesh_x, mesh_y, metadata = (
            generate_checkerboard_mesh_box(
                domain,
                x_list=[4.5],
                y_list=[],
                element_size=10,
                mandatory_coordinates=mandatory,
                return_metadata=True,
            )
        )

        self.assertEqual(mandatory, {"x": [], "y": [2.0, 8.0]})
        np.testing.assert_allclose(mesh_x, [0, 4.5, 10])
        np.testing.assert_allclose(mesh_y, [0, 2, 8, 10])
        self.assertEqual(metadata["grid_shape"], (4, 3))
        self.assertEqual(elements.shape, (6, 4))
        np.testing.assert_allclose(
            metadata["pinned_axis_coordinates"]["x"],
            [0, 10],
        )
        np.testing.assert_allclose(
            metadata["pinned_axis_coordinates"]["y"],
            [0, 2, 8, 10],
        )

        node_pairs = {
            (round(float(x), 8), round(float(y), 8))
            for x, y in nodes[:, :2]
        }
        self.assertIn((4.5, 2.0), node_pairs)
        self.assertIn((4.5, 8.0), node_pairs)

        rail_index = build_structured_rail_node_index(
            metadata["x_nodes"],
            metadata["y_nodes"],
            {"x": [{"coord": 4.5}], "y": []},
        )
        x_rail = rail_index["x"][0]
        np.testing.assert_allclose(x_rail["span_values"], [0, 2, 8, 10])
        span_mask = (
            (x_rail["span_values"] >= 2)
            & (x_rail["span_values"] <= 8)
        )
        endpoint_node_ids = x_rail["node_ids"][span_mask]
        np.testing.assert_allclose(
            nodes[endpoint_node_ids, :2],
            [[4.5, 2], [4.5, 8]],
        )

    def test_horizontal_line_contributes_x_span_endpoints(self):
        faces = [
            {
                "type": "LINE",
                "dim": [7, 3, 2, 3],
                "bottom_z": 0,
                "top_z": 1,
            }
        ]

        self.assertEqual(
            line_span_endpoint_coordinates(faces),
            {"x": [2.0, 7.0], "y": []},
        )

    def test_box_and_polygon_edges_contribute_all_cross_axis_stations(self):
        faces = [
            {
                "type": "BOX",
                "dim": [1, 2, 9, 8],
                "bottom_z": 0,
                "top_z": 1,
            },
            {
                "type": "POLYGON",
                "dim": [
                    [[3, 3], [3, 7], [7, 7], [7, 5], [5, 5], [5, 3]]
                ],
                "bottom_z": 2,
                "top_z": 3,
            },
        ]

        mandatory = feature_span_endpoint_coordinates(faces)

        self.assertEqual(mandatory["x"], [1.0, 3.0, 5.0, 7.0, 9.0])
        self.assertEqual(mandatory["y"], [2.0, 3.0, 5.0, 7.0, 8.0])
        self.assertEqual(
            line_span_endpoint_coordinates(faces),
            mandatory,
        )

    def test_structured_index_reports_missing_rail_without_mesh_scan(self):
        index = build_structured_rail_node_index(
            x_nodes=[0, 1, 2],
            y_nodes=[0, 3],
            rails_by_axis={"x": [1.5], "y": [3]},
        )

        self.assertEqual(len(index["x"][0]["node_ids"]), 0)
        np.testing.assert_array_equal(index["y"][0]["node_ids"], [3, 4, 5])
        np.testing.assert_allclose(index["y"][0]["span_values"], [0, 1, 2])

    def test_box_domain_exposes_only_physical_axis_boundaries_as_pinned(self):
        box = {"type": "BOX", "dim": [0, 1, 10, 11]}
        polygon = {
            "type": "POLYGON",
            "dim": [[[0, 0], [0, 1], [1, 1], [1, 0]]],
        }

        self.assertEqual(
            mesh_domain_pinned_coordinates(box),
            {"x": [0.0, 10.0], "y": [1.0, 11.0]},
        )
        self.assertEqual(
            mesh_domain_pinned_coordinates(polygon),
            {"x": [], "y": []},
        )


class TestMeshGeneratorValidation(unittest.TestCase):
    def test_node_generation_broadcasts_into_final_storage(self):
        """Node generation must not allocate two full meshgrid arrays."""
        x_nodes = np.asarray([0.0, 1.5, 3.0])
        y_nodes = np.asarray([-2.0, 4.0])

        with mock.patch.object(
            mesh_box_module.np,
            "meshgrid",
            side_effect=AssertionError("full meshgrid temporary allocated"),
        ):
            nodes = _mesh_nodes(x_nodes, y_nodes)

        np.testing.assert_array_equal(
            nodes,
            [
                [0.0, -2.0, 0.0],
                [1.5, -2.0, 0.0],
                [3.0, -2.0, 0.0],
                [0.0, 4.0, 0.0],
                [1.5, 4.0, 0.0],
                [3.0, 4.0, 0.0],
            ],
        )

    def test_connectivity_generation_uses_bounded_chunks(self):
        """Only one requested-size vector may accompany final connectivity."""
        nx = 41
        ny = 17
        chunk_size = 37
        arange_lengths = []
        original_arange = np.arange

        def record_arange(start, stop=None, *args, **kwargs):
            result = original_arange(start, stop, *args, **kwargs)
            arange_lengths.append(len(result))
            return result

        with mock.patch.object(
            mesh_box_module.np,
            "arange",
            side_effect=record_arange,
        ):
            elements = _mesh_elements(nx, ny, chunk_size=chunk_size)

        self.assertEqual(elements.dtype, np.int32)
        self.assertEqual(len(elements), (nx - 1) * (ny - 1))
        self.assertTrue(arange_lengths)
        self.assertLessEqual(max(arange_lengths), chunk_size)
        for element_id in (0, 36, 37, len(elements) - 1):
            row, column = divmod(element_id, nx - 1)
            lower_left = row * nx + column
            np.testing.assert_array_equal(
                elements[element_id],
                [
                    lower_left,
                    lower_left + 1,
                    lower_left + nx + 1,
                    lower_left + nx,
                ],
            )

    def test_generator_fails_before_unindexable_grid_allocation(self):
        """A grid outside int32 connectivity capacity must fail closed."""
        with self.assertRaisesRegex(OverflowError, "int32 connectivity"):
            checkerboard_mesh_box(
                [0.0, float(np.iinfo(np.int32).max)],
                [0.0, 1.0],
                element_size=1.0,
            )

    def test_checkerboard_rejects_nonpositive_or_nonfinite_element_size(self):
        for element_size in (0, -1, np.nan, np.inf, True, "1"):
            with self.subTest(element_size=element_size):
                with self.assertRaisesRegex(ValueError, "positive finite"):
                    checkerboard_mesh_box(
                        [0, 1],
                        [0, 1],
                        element_size,
                    )

    def test_checkerboard_rejects_invalid_axis_coordinates(self):
        bad_axes = (
            [0],
            [0, 0],
            [1, 0],
            [0, np.nan],
            [0, np.inf],
        )
        for x_values in bad_axes:
            with self.subTest(x_values=x_values):
                with self.assertRaisesRegex(ValueError, "x_list"):
                    checkerboard_mesh_box(x_values, [0, 1], 1)

    def test_domain_rejects_nonfinite_coordinates(self):
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "finite"):
                    normalize_mesh_domain(
                        {"type": "BOX", "dim": [0, 0, value, 1]}
                    )

    def test_generator_rejects_mandatory_coordinate_outside_domain(self):
        domain = normalize_mesh_domain(
            {"type": "BOX", "dim": [0, 0, 10, 10]}
        )

        with self.assertRaisesRegex(ValueError, "outside"):
            generate_checkerboard_mesh_box(
                domain,
                x_list=[],
                y_list=[],
                element_size=1,
                mandatory_coordinates={"x": [11], "y": []},
            )

    def test_generator_preserves_distinct_coordinates_closer_than_old_eps(self):
        domain = normalize_mesh_domain(
            {"type": "BOX", "dim": [0, 0, 10, 10]}
        )
        first = 2.0
        second = first + 5e-10

        _, _, _, mesh_y = generate_checkerboard_mesh_box(
            domain,
            x_list=[],
            y_list=[],
            element_size=1,
            mandatory_coordinates={"x": [], "y": [first, second]},
        )

        self.assertIn(first, mesh_y)
        self.assertIn(second, mesh_y)
        self.assertEqual(np.count_nonzero(mesh_y == first), 1)
        self.assertEqual(np.count_nonzero(mesh_y == second), 1)
        self.assertTrue(np.all(np.diff(mesh_y) > 0))


if __name__ == "__main__":
    unittest.main()
