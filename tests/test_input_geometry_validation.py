import os
import sys
import unittest

import numpy as np


SRC_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "src")
)
sys.path.insert(0, SRC_ROOT)

from optimal_checkerboard.algorithms.classify_line import _classify_line
from optimal_checkerboard.algorithms.drag import search_face_element
from optimal_checkerboard.algorithms.feature_lines import _extract_lines
from optimal_checkerboard.data_structure.face import Face
from optimal_checkerboard.data_structure.geometry import Obj
from optimal_checkerboard.data_structure.layer import Layer
from optimal_checkerboard.data_structure.mesh import Mesh
from optimal_checkerboard.data_structure.metal import Metal


class TestInputGeometryValidation(unittest.TestCase):
    def test_reversed_box_is_canonical_across_input_and_predicate(self):
        face = Face("BOX", [4, 3, 1, -2])
        self.assertEqual(face.dim, [1.0, -2.0, 4.0, 3.0])

        face.set_position_abs(10, 20)
        self.assertEqual(face.dim_abs, [11.0, 18.0, 14.0, 23.0])

        lines = _extract_lines(
            [
                {
                    "type": "BOX",
                    "dim": [4, 3, 1, -2],
                    "bottom_z": 0,
                    "top_z": 1,
                }
            ]
        )
        self.assertEqual(len(lines), 4)
        self.assertEqual(
            {(line[0][0], line[0][1]) for line in lines},
            {(1.0, -2.0), (4.0, -2.0), (4.0, 3.0), (1.0, 3.0)},
        )

        element_coordinates = np.array(
            [
                [1, -2, 2, -2, 2, -1, 1, -1],
                [4, 3, 5, 3, 5, 4, 4, 4],
            ],
            dtype=float,
        )
        np.testing.assert_array_equal(
            search_face_element(
                element_coordinates,
                "BOX",
                [4, 3, 1, -2],
            ),
            [0],
        )

    def test_nonfinite_vertical_inputs_fail_at_construction(self):
        bad_values = (float("nan"), float("inf"), float("-inf"))
        for value in bad_values:
            with self.subTest(kind="Obj z", value=value):
                with self.assertRaisesRegex(ValueError, "finite"):
                    Obj("BOX", [0, 0, 1, 1], z=value)
            with self.subTest(kind="Metal begin", value=value):
                with self.assertRaisesRegex(ValueError, "finite"):
                    Metal("NORMAL", value, 1, "M")
            with self.subTest(kind="Metal end", value=value):
                with self.assertRaisesRegex(ValueError, "finite"):
                    Metal("NORMAL", 0, value, "M")
            with self.subTest(kind="Mesh begin", value=value):
                with self.assertRaisesRegex(ValueError, "finite"):
                    Mesh(value, 1)
            with self.subTest(kind="Mesh end", value=value):
                with self.assertRaisesRegex(ValueError, "finite"):
                    Mesh(0, value)
            with self.subTest(kind="Layer thickness", value=value):
                with self.assertRaisesRegex(ValueError, "thickness"):
                    Layer("M", value)

    def test_nonfinite_absolute_z_offsets_fail_before_propagation(self):
        obj = Obj("BOX", [0, 0, 1, 1])
        metal = Metal("NORMAL", 0, 1, "M")
        mesh = Mesh(0, 1)
        face = Face("BOX", [0, 0, 1, 1])

        for item in (obj, metal, mesh, face):
            with self.subTest(item=type(item).__name__):
                with self.assertRaisesRegex(ValueError, "finite"):
                    item.set_position_abs(z=float("nan"))

    def test_feature_and_classification_reject_nonfinite_values(self):
        for key in ("bottom_z", "top_z"):
            face = {
                "type": "LINE",
                "dim": [0, 0, 1, 0],
                "bottom_z": 0,
                "top_z": 1,
            }
            face[key] = float("nan")
            with self.subTest(face_key=key):
                with self.assertRaisesRegex(ValueError, "finite"):
                    _extract_lines([face])

        for line in (
            [[float("nan"), 0], [1, 0], [0, 1]],
            [[0, 0], [1, 0], [0, float("inf")]],
            [[0, 0, float("nan")], [1, 0, float("nan")]],
        ):
            with self.subTest(line=line):
                with self.assertRaisesRegex(ValueError, "finite"):
                    _classify_line([line])

    def test_nonfinite_face_coordinates_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            Face("BOX", [0, 0, float("nan"), 1])
        with self.assertRaisesRegex(ValueError, "finite"):
            search_face_element(
                np.array([[0, 0, 1, 0, 1, 1, 0, 1]], dtype=float),
                "BOX",
                [0, 0, float("inf"), 1],
            )


if __name__ == "__main__":
    unittest.main()
