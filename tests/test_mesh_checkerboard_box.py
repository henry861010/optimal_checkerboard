import os
import sys
import unittest

SRC_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "src")
)
sys.path.insert(0, SRC_ROOT)

import numpy as np

from optimal_checkerboard import ELEMENT_2D_VOLUMN, OptimalMesh25D


class TestMeshCheckerboardBox(unittest.TestCase):
    def test_mesh_checkerboard_box_requires_pattern(self):
        """Verify mesh generation requires set_pattern first."""
        mesher = OptimalMesh25D()

        with self.assertRaisesRegex(RuntimeError, "set_pattern"):
            mesher.mesh_checkerboard_box([0, 0, 10, 10])

    def test_mesh_checkerboard_box_rejects_invalid_dim(self):
        """Verify checkerboard bounds must be 4D or 6D."""
        faces = [{"type": "BOX", "dim": [0, 0, 0, 10, 10, 0]}]

        mesher = OptimalMesh25D()
        mesher.set_pattern(faces, element_size=10, ratio=0.1)

        with self.assertRaisesRegex(ValueError, "dim must be"):
            mesher.mesh_checkerboard_box([0, 0, 10])

    def test_mesh_checkerboard_box_builds_element_internal(self):
        """Verify generated meshes include the drag-time element table."""
        faces = [{"type": "BOX", "dim": [0, 0, 0, 10, 10, 0]}]

        mesher = OptimalMesh25D()
        mesher.set_pattern(faces, element_size=10, ratio=0.1)
        mesh = mesher.mesh_checkerboard_box([0, 0, 10, 10])

        self.assertEqual(mesh.element_internal.shape, (1, 10))
        self.assertAlmostEqual(
            float(mesh.element_internal[0, ELEMENT_2D_VOLUMN]),
            100.0,
        )
        np.testing.assert_allclose(
            mesh.element_internal[0, 2:10],
            np.asarray([0, 0, 10, 0, 10, 10, 0, 10], dtype=np.float32),
        )


if __name__ == "__main__":
    unittest.main()
