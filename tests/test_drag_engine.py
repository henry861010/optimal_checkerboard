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
