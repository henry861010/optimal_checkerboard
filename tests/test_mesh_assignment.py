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

        with self.assertRaisesRegex(ValueError, "required x-axis shared rail"):
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

        with self.assertRaisesRegex(ValueError, "snap rule span"):
            mesher.mesh_assignment(mesh2d)


if __name__ == "__main__":
    unittest.main()
