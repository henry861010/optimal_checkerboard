import os
import sys
import unittest

SRC_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "src")
)
sys.path.insert(0, SRC_ROOT)

from optimal_checkerboard.data_structure.face import Face
from optimal_checkerboard.data_structure.geometry import Obj
from optimal_checkerboard.data_structure.mesh import Mesh
from optimal_checkerboard.data_structure.metal import Metal


class TestDataStructureInfo(unittest.TestCase):
    def test_face_info_returns_local_or_absolute_coordinates(self):
        face = Face("BOX", [0, 0, 2, 3])
        face.set_position_abs(10, 20)

        self.assertEqual(
            face.info(),
            {
                "type": "BOX",
                "dim": [0, 0, 2, 3],
                "dim_abs": [10, 20, 12, 23],
            },
        )
        self.assertEqual(
            face.info(isAbs=True),
            {
                "type": "BOX",
                "dim": [10, 20, 12, 23],
            },
        )

    def test_metal_info_uses_is_abs_and_serializes_faces(self):
        metal = Metal(
            "NORMAL",
            begin=1,
            end=3,
            material="M1",
            ranges=[Face("BOX", [0, 0, 2, 2])],
        )
        metal.set_position_abs(10, 20, 30)

        local_info = metal.info()
        absolute_info = metal.info(isAbs=True)

        self.assertEqual(local_info["begin"], 1)
        self.assertEqual(local_info["begin_abs"], 31)
        self.assertEqual(local_info["ranges"][0]["dim"], [0, 0, 2, 2])
        self.assertEqual(
            local_info["ranges"][0]["dim_abs"],
            [10, 20, 12, 22],
        )
        self.assertEqual(absolute_info["begin"], 31)
        self.assertEqual(absolute_info["end"], 33)
        self.assertEqual(
            absolute_info["ranges"][0]["dim"],
            [10, 20, 12, 22],
        )

    def test_obj_and_mesh_info_serialize_nested_faces(self):
        obj = Obj("BOX", [0, 0, 10, 10], z=2)
        obj.add_layer(thk=4, material="BASE")
        obj.meshs.append(
            Mesh(
                begin=1,
                end=3,
                face=Face("BOX", [2, 2, 4, 4]),
            )
        )
        obj.set_position_abs(5, 6, 7)

        local_info = obj.info()
        absolute_info = obj.info(isAbs=True)

        self.assertEqual(local_info["face"]["dim"], [0, 0, 10, 10])
        self.assertEqual(local_info["face"]["dim_abs"], [5, 6, 15, 16])
        self.assertEqual(
            local_info["meshs"][0]["face"]["dim"],
            [2, 2, 4, 4],
        )
        self.assertEqual(absolute_info["z_abs"], 9)
        self.assertEqual(absolute_info["face"]["dim"], [5, 6, 15, 16])
        self.assertEqual(
            absolute_info["meshs"][0]["face"]["dim"],
            [7, 8, 9, 10],
        )


if __name__ == "__main__":
    unittest.main()
