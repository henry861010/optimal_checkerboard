import os
import sys
import unittest

SRC_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "src")
)
sys.path.insert(0, SRC_ROOT)

import numpy as np

from optimal_checkerboard import OptimalMesh25D


class TestMeshCheckerboardBox(unittest.TestCase):
    def test_mesh_checkerboard_box_requires_pattern(self):
        """Verify mesh generation requires set_pattern first."""
        mesher = OptimalMesh25D()

        with self.assertRaisesRegex(RuntimeError, "set_pattern"):
            mesher.mesh_checkerboard_box([0, 0, 10, 10])

    def test_mesh_checkerboard_box_rejects_invalid_dim(self):
        """Verify checkerboard bounds must be 4D or 6D."""
        faces = [
            {
                "type": "BOX",
                "dim": [0, 0, 10, 10],
                "bottom_z": 0,
                "top_z": 0,
            }
        ]

        mesher = OptimalMesh25D()
        mesher.set_pattern(faces, element_size=10, ratio=0.1)

        with self.assertRaisesRegex(ValueError, "dim must be"):
            mesher.mesh_checkerboard_box([0, 0, 10])

    def test_mesh_checkerboard_box_builds_nodes_and_elements(self):
        """Verify generated meshes expose only node coordinates and elements."""
        faces = [
            {
                "type": "BOX",
                "dim": [0, 0, 10, 10],
                "bottom_z": 0,
                "top_z": 0,
            }
        ]

        mesher = OptimalMesh25D()
        mesher.set_pattern(faces, element_size=10, ratio=0.1)
        mesh = mesher.mesh_checkerboard_box([0, 0, 10, 10])

        self.assertFalse(hasattr(mesh, "element_internal"))
        self.assertEqual(mesh.nodes.shape, (4, 3))
        self.assertEqual(mesh.elements.shape, (1, 4))
        np.testing.assert_allclose(
            mesh.nodes[mesh.elements[0], :2].reshape(8),
            np.asarray([0, 0, 10, 0, 10, 10, 0, 10], dtype=np.float32),
        )
        self.assertIsNotNone(mesher.rail_node_index)


if __name__ == "__main__":
    unittest.main()
