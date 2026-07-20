"""Object geometry data structure for hierarchical 2.5D layouts.

An Obj combines a face, layers, metals, mesh controls, and child objects, then
propagates absolute coordinates through the hierarchy from parent origins.
"""

from copy import deepcopy
import math

from .face import Face
from .layer import Layer
from .metal import Metal
from .mesh import Mesh


class Obj:
    """Represent an object with local geometry and nested child objects."""

    def __init__(self, type, dim, z=0):
        self.z = _finite_z(z, "Obj z")
        self.face = Face(type=type, dim=dim)
        self.layers: list[Layer] = []
        self.metals: list[Metal] = []
        self.meshs: list[Mesh] = []
        self.child_objs: list["Obj"] = []
        self.parent_obj: "Obj" = None

        self.z_abs = None

    @property
    def thk(self):
        """Return live stack thickness so layer edits cannot stale a cache."""
        total = 0.0
        for layer in self.layers:
            thickness = float(layer.thk)
            if not math.isfinite(thickness) or thickness <= 0.0:
                raise ValueError("Every layer thickness must be finite and positive")
            total += thickness
        return total

    def set_position_abs(self, x=0, y=0, z=0):
        x = _finite_z(x, "Obj x offset")
        y = _finite_z(y, "Obj y offset")
        z = _finite_z(z, "Obj z offset")
        self.face.set_position_abs(x, y, z)
        self.z_abs = _finite_z(self.z + z, "Obj z_abs")

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

    def info(self, isAbs=False):
        if isAbs:
            return {
                "z": self.z,
                "z_abs": self.z_abs,
                "face": self.face.info(isAbs=True),
                "thk": self.thk,
                "layers": [layer.info(isAbs=True) for layer in self.layers],
                "metals": [metal.info(isAbs=True) for metal in self.metals],
                "meshs": [mesh.info(isAbs=True) for mesh in self.meshs],
                "child_objs": [
                    child_obj.info(isAbs=True) for child_obj in self.child_objs
                ],
            }
        else:
            return {
                "z": self.z,
                "z_abs": self.z_abs,
                "face": self.face.info(),
                "thk": self.thk,
                "layers": [layer.info() for layer in self.layers],
                "metals": [metal.info() for metal in self.metals],
                "meshs": [mesh.info() for mesh in self.meshs],
                "child_objs": [child_obj.info() for child_obj in self.child_objs],
            }

    def add_layer(self, thk, material):
        thickness = float(thk)
        if not math.isfinite(thickness) or thickness <= 0.0:
            raise ValueError("Layer thickness must be finite and greater than zero")
        self.layers.append(Layer(thk=thickness, material=material))

    def add_child(self, child_obj: "Obj"):
        if not isinstance(child_obj, Obj):
            raise TypeError("child_obj must be an Obj")

        # Inspect the candidate subtree by identity.  This catches self-links,
        # ancestor links, and already-corrupt candidate cycles before mutation.
        pending = [child_obj]
        visited = set()
        while pending:
            candidate = pending.pop()
            candidate_id = id(candidate)
            if candidate_id in visited:
                raise ValueError("child hierarchy contains a cycle or alias")
            visited.add(candidate_id)
            if candidate is self:
                raise ValueError("Adding child_obj would create a hierarchy cycle")
            pending.extend(candidate.child_objs)

        if child_obj.parent_obj is not None:
            raise ValueError("child_obj already belongs to a parent")
        if any(existing is child_obj for existing in self.child_objs):
            raise ValueError("child_obj is already attached to this parent")

        child_obj.parent_obj = self
        self.child_objs.append(child_obj)


def _finite_z(value, name):
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value
