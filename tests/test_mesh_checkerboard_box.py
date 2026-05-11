import os
import sys
import unittest

SRC_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "src")
)
sys.path.insert(0, SRC_ROOT)

import numpy as np

from optimal_checkerboard import OptimalMesh25D
from optimal_checkerboard.data_structure.geometry import Obj


class TestMeshCheckerboardBox(unittest.TestCase):
    def test_mesh_checkerboard_box_requires_pattern(self):
        """Verify mesh generation requires pattern preprocessing first."""
        mesher = OptimalMesh25D()

        with self.assertRaisesRegex(RuntimeError, "set_pattern"):
            mesher.mesh_checkerboard_box()

    def test_mesh_checkerboard_box_requires_root_boundary(self):
        """Verify raw pattern callers need an explicit mesh domain."""
        faces = [
            {
                "type": "BOX",
                "dim": [0, 0, 10, 10],
                "bottom_z": 0,
                "top_z": 0,
            }
        ]

        mesher = OptimalMesh25D()
        mesher._set_pattern(faces, element_size=10, ratio=0.1)

        with self.assertRaisesRegex(RuntimeError, "root mesh boundary"):
            mesher.mesh_checkerboard_box()

    def test_mesh_checkerboard_box_rejects_legacy_dim_argument(self):
        """Verify BOX mesh generation no longer accepts boundary dims."""
        obj = Obj("BOX", [0, 0, 10, 10])

        mesher = OptimalMesh25D()
        mesher.set_pattern_obj(obj, element_size=10, ratio=0.1)

        with self.assertRaises(TypeError):
            mesher.mesh_checkerboard_box([0, 0, 10, 10])

    def test_mesh_checkerboard_box_builds_nodes_and_elements(self):
        """Verify generated meshes expose only node coordinates and elements."""
        obj = Obj("BOX", [0, 0, 10, 10])

        mesher = OptimalMesh25D()
        mesher.set_pattern_obj(obj, element_size=10, ratio=0.1)
        mesh = mesher.mesh_checkerboard_box()

        self.assertFalse(hasattr(mesh, "element_internal"))
        self.assertEqual(mesh.nodes.shape, (4, 3))
        self.assertEqual(mesh.elements.shape, (1, 4))
        np.testing.assert_allclose(
            mesh.nodes[mesh.elements[0], :2].reshape(8),
            np.asarray([0, 0, 10, 0, 10, 10, 0, 10], dtype=np.float32),
        )
        self.assertIsNotNone(mesher.rail_node_index)

    def test_mesh_checkerboard_dispatches_cylinder_to_future_generator(self):
        """Verify CYLINDER domains are routed separately from BOX meshing."""
        obj = Obj("CYLINDER", [0, 0, 5])

        mesher = OptimalMesh25D()
        with self.assertWarnsRegex(RuntimeWarning, "CYLINDER"):
            mesher.set_pattern_obj(obj, element_size=10, ratio=0.1)

        with self.assertRaisesRegex(NotImplementedError, "CYLINDER"):
            mesher.mesh_checkerboard()


if __name__ == "__main__":
    unittest.main()
