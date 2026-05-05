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
    def test_mesh_assignment_converts_nodes_elements_to_element_internal(self):
        """Verify assigned meshes are converted once at API boundary."""
        faces = [{"type": "BOX", "dim": [0, 0, 0, 10, 10, 0]}]
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
        mesher.set_pattern(faces, element_size=10, ratio=0.1)
        assigned = mesher.mesh_assignment(mesh2d)

        self.assertIs(assigned, mesh2d)
        self.assertEqual(assigned.element_internal.shape, (1, 10))
        self.assertIsNotNone(mesher.rail_reference_index)

    def test_mesh_assignment_preserves_existing_element_internal(self):
        """Verify caller-provided internal tables are reused when valid."""
        faces = [{"type": "BOX", "dim": [0, 0, 0, 10, 10, 0]}]
        element_internal = np.asarray(
            [[0, 100, 0, 0, 10, 0, 10, 10, 0, 10]],
            dtype=np.float32,
        )
        mesh2d = Mesh2D(
            nodes=np.empty((0, 3), dtype=np.float32),
            elements=np.empty((0, 4), dtype=np.int32),
            element_internal=element_internal,
        )

        mesher = OptimalMesh25D()
        mesher.set_pattern(faces, element_size=10, ratio=0.1)
        mesher.mesh_assignment(mesh2d)

        self.assertIs(mesh2d.element_internal, element_internal)
        np.testing.assert_allclose(mesh2d.element_internal, element_internal)

    def test_mesh_assignment_rejects_invalid_node_shape(self):
        """Verify assigned mesh nodes must be a 2D coordinate array."""
        faces = [{"type": "BOX", "dim": [0, 0, 0, 10, 10, 0]}]
        mesh2d = Mesh2D(
            nodes=np.asarray([0, 0, 0], dtype=np.float32),
            elements=np.asarray([[0, 1, 2, 3]], dtype=np.int32),
        )

        mesher = OptimalMesh25D()
        mesher.set_pattern(faces, element_size=10, ratio=0.1)

        with self.assertRaisesRegex(ValueError, "mesh2d.nodes"):
            mesher.mesh_assignment(mesh2d)


if __name__ == "__main__":
    unittest.main()
