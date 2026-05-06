import os
import sys
import unittest

SRC_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "src")
)
sys.path.insert(0, SRC_ROOT)

import numpy as np

from optimal_checkerboard.algorithms.drag import Engin25D


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


class TestDragEngine(unittest.TestCase):
    def test_set_2d_uses_split_element_arrays(self):
        mesh = SimpleMesh2D()
        engine = Engin25D()

        engine.set_2D(mesh)

        self.assertFalse(hasattr(engine, "element_internal"))
        np.testing.assert_array_equal(engine.element_2D, mesh.elements)
        np.testing.assert_allclose(engine.element_2D_volumn, [1.0, 1.0])
        np.testing.assert_array_equal(engine.element_2D_comp, [0, 0])

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


if __name__ == "__main__":
    unittest.main()
