"""Object geometry data structure for hierarchical 2.5D layouts.

An Obj combines a face, layers, metals, mesh controls, and child objects, then
propagates absolute coordinates through the hierarchy from parent origins.
"""

from copy import deepcopy

from .face import Face
from .layer import Layer
from .metal import Metal
from .mesh import Mesh


class Obj:
    """Represent an object with local geometry and nested child objects."""

    def __init__(self, type, dim, z=0):
        self.z = z
        self.face = Face(type=type, dim=dim)
        self.layers: list[Layer] = []
        self.metals: list[Metal] = []
        self.meshs: list[Mesh] = []
        self.child_objs: list["Obj"] = []
        self.parent_obj: "Obj" = None

        self.thk = 0

        self.z_abs = None

    def set_position_abs(self, x=0, y=0, z=0):
        self.face.set_position_abs(x, y, z)
        self.z_abs = self.z + z

        if self.face.type in ("BOX", "CYLINDER"):
            x_ref = self.face.dim_abs[0]
            y_ref = self.face.dim_abs[1]
            z_ref = self.z_abs
        elif self.face.type == "POLYGON":
            x_ref = self.face.dim_abs[0][0][0]
            y_ref = self.face.dim_abs[0][0][1]
            z_ref = self.z_abs
        else:
            raise ValueError("unknown type")

        for metal in self.metals:
            metal.set_position_abs(x_ref, y_ref, z_ref)

        for mesh in self.meshs:
            mesh.set_position_abs(x_ref, y_ref, z_ref)

        for child_obj in self.child_objs:
            child_obj.set_position_abs(x_ref, y_ref, z_ref)

    def copy(self):
        obj_dup = Obj(self.face.type, deepcopy(self.face.dim), self.z)

        obj_dup.z = self.z
        obj_dup.z_abs = self.z_abs
        obj_dup.thk = self.thk
        obj_dup.face = self.face.copy()
        obj_dup.layers = [layer.copy() for layer in self.layers]
        obj_dup.metals = [metal.copy() for metal in self.metals]
        obj_dup.meshs = [mesh.copy() for mesh in self.meshs]
        obj_dup.child_objs = []
        for child_obj in self.child_objs:
            child_obj_dup = child_obj.copy()
            child_obj_dup.parent_obj = obj_dup
            obj_dup.child_objs.append(child_obj_dup)
        obj_dup.parent_obj = None

        return obj_dup

    def dict(self):
        return {
            "z": self.z,
            "z_abs": self.z_abs,
            "face": self.face.dict(),
            "thk": self.thk,
            "layers": [layer.dict() for layer in self.layers],
            "metals": [metal.dict() for metal in self.metals],
            "meshs": [mesh.dict() for mesh in self.meshs],
            "child_objs": [child_obj.dict() for child_obj in self.child_objs],
        }

    def dict_abs(self):
        return {
            "z": self.z,
            "z_abs": self.z_abs,
            "face": self.face.dict_abs(),
            "thk": self.thk,
            "layers": [layer.dict_abs() for layer in self.layers],
            "metals": [metal.dict_abs() for metal in self.metals],
            "meshs": [mesh.dict_abs() for mesh in self.meshs],
            "child_objs": [
                child_obj.dict_abs() for child_obj in self.child_objs
            ],
        }

    def add_layer(self, thk, material):
        self.layers.append(Layer(thk=thk, material=material))
        self.thk += thk

    def add_child(self, child_obj: "Obj"):
        child_obj.parent_obj = self
        self.child_objs.append(child_obj)
