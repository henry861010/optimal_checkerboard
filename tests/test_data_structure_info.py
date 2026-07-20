import os
import sys
import unittest

SRC_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "src")
)
sys.path.insert(0, SRC_ROOT)

from optimal_checkerboard.data_structure.face import Face
from optimal_checkerboard.data_structure.geometry import Obj
from optimal_checkerboard.data_structure.layer import Layer
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

    def test_obj_thickness_is_recomputed_from_current_layers(self):
        obj = Obj("BOX", [0, 0, 1, 1])
        obj.add_layer(2, "A")
        obj.layers.append(Layer("B", 3))
        self.assertEqual(obj.thk, 5)

        obj.layers[0].thk = 7

        self.assertEqual(obj.thk, 10)
        self.assertEqual(obj.copy().thk, 10)

    def test_obj_rejects_invalid_layer_thickness(self):
        obj = Obj("BOX", [0, 0, 1, 1])
        for thickness in (0, -1, float("nan"), float("inf")):
            with self.subTest(thickness=thickness):
                with self.assertRaisesRegex(ValueError, "thickness"):
                    obj.add_layer(thickness, "A")

    def test_obj_child_has_single_parent_and_no_duplicates(self):
        first_parent = Obj("BOX", [0, 0, 1, 1])
        second_parent = Obj("BOX", [0, 0, 1, 1])
        child = Obj("BOX", [0, 0, 1, 1])
        first_parent.add_child(child)

        with self.assertRaisesRegex(ValueError, "parent"):
            second_parent.add_child(child)
        with self.assertRaisesRegex(ValueError, "parent"):
            first_parent.add_child(child)

        self.assertIs(child.parent_obj, first_parent)
        self.assertEqual(len(first_parent.child_objs), 1)
        self.assertEqual(len(second_parent.child_objs), 0)

    def test_obj_child_rejects_cycles_without_partial_mutation(self):
        root = Obj("BOX", [0, 0, 1, 1])
        child = Obj("BOX", [0, 0, 1, 1])
        grandchild = Obj("BOX", [0, 0, 1, 1])
        root.add_child(child)
        child.add_child(grandchild)

        with self.assertRaisesRegex(ValueError, "cycle"):
            grandchild.add_child(root)
        with self.assertRaisesRegex(ValueError, "cycle"):
            root.add_child(root)

        self.assertIsNone(root.parent_obj)
        self.assertEqual(grandchild.child_objs, [])


if __name__ == "__main__":
    unittest.main()
