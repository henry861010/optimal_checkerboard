import os
import sys
import unittest

SRC_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "src")
)
sys.path.insert(0, SRC_ROOT)

import numpy as np

from optimal_checkerboard import Mesh2D, OptimalMesh25D


class TestMeshAssignment(unittest.TestCase):
    @staticmethod
    def _domain_mesher(size=3.0, element_size=1.0):
        mesher = OptimalMesh25D()
        mesher._set_pattern(
            [{
                "type": "BOX",
                "dim": [0.0, 0.0, size, size],
                "bottom_z": 0.0,
                "top_z": 1.0,
            }],
            element_size=element_size,
            ratio=0.0,
            mesh_domain={"type": "BOX", "dim": [0.0, 0.0, size, size]},
        )
        return mesher

    @staticmethod
    def _regular_grid(size=3):
        x_grid, y_grid = np.meshgrid(
            np.arange(size + 1, dtype=np.float64),
            np.arange(size + 1, dtype=np.float64),
        )
        nodes = np.column_stack((x_grid.ravel(), y_grid.ravel()))
        node_ids = np.arange((size + 1) ** 2).reshape(size + 1, size + 1)
        elements = np.column_stack(
            (
                node_ids[:-1, :-1].ravel(),
                node_ids[:-1, 1:].ravel(),
                node_ids[1:, 1:].ravel(),
                node_ids[1:, :-1].ravel(),
            )
        ).astype(np.int32)
        return nodes, elements

    def test_custom_box_mesh_rejects_missing_interior_cell(self):
        nodes, elements = self._regular_grid(size=3)
        elements = np.delete(elements, 4, axis=0)
        mesh = Mesh2D(nodes=nodes, elements=elements)
        mesher = self._domain_mesher(size=3.0)

        with self.assertRaisesRegex(ValueError, "hole|unmatched"):
            mesher.mesh_assignment(mesh)

    def test_custom_box_mesh_rejects_active_elements_outside_domain(self):
        nodes = np.asarray(
            [
                [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0],
                [2.0, 0.0], [3.0, 0.0], [3.0, 1.0], [2.0, 1.0],
            ]
        )
        elements = np.asarray(
            [[0, 1, 2, 3], [4, 5, 6, 7]],
            dtype=np.int32,
        )
        mesh = Mesh2D(nodes=nodes, elements=elements)
        mesher = self._domain_mesher(size=1.0)

        with self.assertRaisesRegex(ValueError, "outside the BOX"):
            mesher.mesh_assignment(mesh)

    def test_custom_box_mesh_rejects_duplicate_elements(self):
        nodes = np.asarray(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
        )
        elements = np.asarray(
            [[0, 1, 2, 3], [0, 1, 2, 3]],
            dtype=np.int32,
        )
        mesh = Mesh2D(nodes=nodes, elements=elements)
        mesher = self._domain_mesher(size=1.0)

        with self.assertRaisesRegex(ValueError, "boundary|duplicate"):
            mesher.mesh_assignment(mesh)

    def test_verified_structured_metadata_uses_compact_assignment_path(self):
        source = self._domain_mesher(size=3.0)
        generated = source.mesh_checkerboard()
        target = self._domain_mesher(size=3.0)

        target.mesh_assignment(generated)

        self.assertEqual(target._structured_grid_shape, (4, 4))
        self.assertIsNone(target.node_axis_rail_ids)
        self.assertIsNone(target._node_element_offsets)
        self.assertIsNone(target._node_element_ids)

    def test_structured_metadata_rejects_tampered_row_major_topology(self):
        source = self._domain_mesher(size=2.0)
        generated = source.mesh_checkerboard()
        generated.elements = generated.elements.copy()
        generated.elements[[0, 1]] = generated.elements[[1, 0]]
        target = self._domain_mesher(size=2.0)

        with self.assertRaisesRegex(ValueError, "row-major"):
            target.mesh_assignment(generated)

        self.assertIsNone(target.mesh2d)

    def test_custom_polygon_domain_is_rejected_until_partition_is_proven(self):
        mesher = OptimalMesh25D()
        mesher._set_pattern(
            [{
                "type": "BOX",
                "dim": [0.0, 0.0, 1.0, 1.0],
                "bottom_z": 0.0,
                "top_z": 1.0,
            }],
            element_size=1.0,
            ratio=0.0,
            mesh_domain={
                "type": "POLYGON",
                "dim": [[
                    [0.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0],
                    [1.0, 0.0],
                ]],
            },
        )
        mesh = Mesh2D(
            nodes=np.asarray(
                [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
            ),
            elements=np.asarray([[0, 1, 2, 3]], dtype=np.int32),
        )

        with self.assertRaisesRegex(NotImplementedError, "only for BOX"):
            mesher.mesh_assignment(mesh)

    def test_mesh_assignment_indexes_nodes_and_elements_directly(self):
        """Verify assigned meshes keep public node/element arrays as source."""
        faces = [
            {
                "type": "BOX",
                "dim": [0, 0, 10, 10],
                "bottom_z": 0,
                "top_z": 0,
            }
        ]
        mesh2d = Mesh2D(
            nodes=np.asarray(
                [
                    [0, 0, 0],
                    [10, 0, 0],
                    [10, 10, 0],
                    [0, 10, 0],
                ],
                dtype=np.float32,
            ),
            elements=np.asarray([[0, 1, 2, 3]], dtype=np.int32),
        )

        mesher = OptimalMesh25D()
        mesher._set_pattern(faces, element_size=10, ratio=0.1)
        assigned = mesher.mesh_assignment(mesh2d)

        self.assertIs(assigned, mesh2d)
        self.assertFalse(hasattr(assigned, "element_internal"))
        np.testing.assert_array_equal(assigned.elements, mesh2d.elements)
        np.testing.assert_allclose(assigned.nodes, mesh2d.nodes)
        self.assertIsNotNone(mesher.rail_node_index)
        self.assertIsNotNone(mesher.rail_reference_index)

    def test_mesh_assignment_normalizes_integer_connectivity_to_int32(self):
        """Valid custom connectivity has one compact execution dtype."""
        for dtype in (np.int16, np.int64, np.uint32):
            with self.subTest(dtype=dtype):
                nodes, elements = self._regular_grid(size=1)
                mesh2d = Mesh2D(
                    nodes=nodes,
                    elements=elements.astype(dtype),
                )
                mesher = self._domain_mesher(size=1.0)

                mesher.mesh_assignment(mesh2d)

                self.assertEqual(mesh2d.elements.dtype, np.int32)
                np.testing.assert_array_equal(mesh2d.elements, elements)

    def test_mesh_assignment_rejects_connectivity_above_int32_capacity(self):
        """Oversized ids must fail before a narrowing integer conversion."""
        nodes, _ = self._regular_grid(size=1)
        original_elements = np.asarray(
            [[0, 1, 2, np.iinfo(np.int32).max + 1]],
            dtype=np.uint64,
        )
        mesh2d = Mesh2D(nodes=nodes, elements=original_elements)
        mesher = self._domain_mesher(size=1.0)

        with self.assertRaisesRegex(OverflowError, "int32 connectivity"):
            mesher.mesh_assignment(mesh2d)

        self.assertIs(mesh2d.elements, original_elements)
        self.assertEqual(mesh2d.elements.dtype, np.uint64)

    def test_mesh_assignment_rejects_out_of_bounds_element_ids(self):
        """Verify element connectivity must reference existing nodes."""
        faces = [
            {
                "type": "BOX",
                "dim": [0, 0, 10, 10],
                "bottom_z": 0,
                "top_z": 0,
            }
        ]
        mesh2d = Mesh2D(
            nodes=np.asarray(
                [
                    [0, 0, 0],
                    [10, 0, 0],
                    [10, 10, 0],
                ],
                dtype=np.float32,
            ),
            elements=np.asarray([[0, 1, 2, 3]], dtype=np.int32),
        )

        mesher = OptimalMesh25D()
        mesher._set_pattern(faces, element_size=10, ratio=0.1)

        with self.assertRaisesRegex(ValueError, "outside mesh2d.nodes"):
            mesher.mesh_assignment(mesh2d)

    def test_mesh_assignment_rejects_invalid_node_shape(self):
        """Verify assigned mesh nodes must be a 2D coordinate array."""
        faces = [
            {
                "type": "BOX",
                "dim": [0, 0, 10, 10],
                "bottom_z": 0,
                "top_z": 0,
            }
        ]
        mesh2d = Mesh2D(
            nodes=np.asarray([0, 0, 0], dtype=np.float32),
            elements=np.asarray([[0, 1, 2, 3]], dtype=np.int32),
        )

        mesher = OptimalMesh25D()
        mesher._set_pattern(faces, element_size=10, ratio=0.1)

        with self.assertRaisesRegex(ValueError, "mesh2d.nodes"):
            mesher.mesh_assignment(mesh2d)

    def test_mesh_assignment_rejects_missing_required_rail(self):
        """Verify every required shared rail has active mesh nodes."""
        faces = [
            {
                "type": "LINE",
                "dim": [1, 0, 1, 5],
                "bottom_z": 0,
                "top_z": 10,
            },
            {
                "type": "LINE",
                "dim": [1.5, 6, 1.5, 11],
                "bottom_z": 20,
                "top_z": 30,
            },
        ]
        mesh2d = Mesh2D(
            nodes=np.asarray(
                [[0, 0], [3, 0], [3, 11], [0, 11]],
                dtype=np.float64,
            ),
            elements=np.asarray([[0, 1, 2, 3]], dtype=np.int32),
        )
        mesher = OptimalMesh25D()
        mesher._set_pattern(faces, element_size=5, ratio=0.2)

        with self.assertRaisesRegex(ValueError, "no collinear edge chain"):
            mesher.mesh_assignment(mesh2d)

    def test_mesh_assignment_rejects_missing_rule_span_nodes(self):
        """Verify a present rail must cover each required snap-rule span."""
        faces = [
            {
                "type": "LINE",
                "dim": [1, 0, 1, 5],
                "bottom_z": 0,
                "top_z": 10,
            },
            {
                "type": "LINE",
                "dim": [1.5, 6, 1.5, 11],
                "bottom_z": 20,
                "top_z": 30,
            },
        ]
        mesh2d = Mesh2D(
            nodes=np.asarray(
                [
                    [0, 0],
                    [1.25, 0],
                    [3, 0],
                    [0, 5],
                    [1.25, 5],
                    [3, 5],
                ],
                dtype=np.float64,
            ),
            elements=np.asarray(
                [[0, 1, 4, 3], [1, 2, 5, 4]],
                dtype=np.int32,
            ),
        )
        mesher = OptimalMesh25D()
        mesher._set_pattern(faces, element_size=5, ratio=0.2)

        with self.assertRaisesRegex(ValueError, "missing a feature endpoint"):
            mesher.mesh_assignment(mesh2d)

    def test_exact_nearby_rails_never_alias_during_assignment_reset(self):
        """Distinct sub-micron rails must retain a positive-area cell."""
        width = 5e-7
        faces = [
            {
                "type": "BOX",
                "dim": [0.0, 0.0, width, 1.0],
                "bottom_z": 0.0,
                "top_z": 1.0,
            }
        ]
        original_nodes = np.asarray(
            [[0.0, 0.0], [width, 0.0], [width, 1.0], [0.0, 1.0]],
            dtype=np.float64,
        )
        mesh2d = Mesh2D(
            nodes=original_nodes.copy(),
            elements=np.asarray([[0, 1, 2, 3]], dtype=np.int32),
        )

        mesher = OptimalMesh25D()
        mesher._set_pattern(faces, element_size=1.0, ratio=0.0)
        mesher.mesh_assignment(mesh2d)

        np.testing.assert_array_equal(mesh2d.nodes, original_nodes)
        self.assertGreater(mesher._signed_quad_areas(mesh2d.nodes, [0])[0], 0)

    def test_float32_custom_nodes_are_promoted_before_close_target_snap(self):
        """Low precision storage must not silently round away a pattern."""
        first_target = 0.99999999
        second_target = 1.00000001
        faces = [
            {
                "type": "LINE",
                "dim": [first_target, 0.0, first_target, 1.0],
                "bottom_z": 0.0,
                "top_z": 0.0,
            },
            {
                "type": "LINE",
                "dim": [second_target, 0.0, second_target, 1.0],
                "bottom_z": 2.0,
                "top_z": 2.0,
            },
        ]
        mesh2d = Mesh2D(
            nodes=np.asarray(
                [
                    [0.0, 0.0], [1.0, 0.0], [2.0, 0.0],
                    [0.0, 1.0], [1.0, 1.0], [2.0, 1.0],
                ],
                dtype=np.float32,
            ),
            elements=np.asarray(
                [[0, 1, 4, 3], [1, 2, 5, 4]],
                dtype=np.int32,
            ),
        )
        mesher = OptimalMesh25D()
        mesher._set_pattern(faces, element_size=1.0, ratio=0.1)
        mesher.mesh_assignment(mesh2d)

        mesher.apply_snap_rules_at_z(2.0)

        self.assertEqual(mesh2d.nodes.dtype, np.float64)
        rail_nodes = mesher.rail_node_index["x"][0]["node_ids"]
        self.assertTrue(np.all(mesh2d.nodes[rail_nodes, 0] == second_target))


if __name__ == "__main__":
    unittest.main()
