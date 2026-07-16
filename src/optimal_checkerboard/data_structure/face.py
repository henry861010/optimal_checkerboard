"""Face data structure for two-dimensional layout primitives.

A Face stores local dimensions for boxes, cylinders, and polygons, then
derives absolute coordinates when placed under an object origin.
"""

from copy import deepcopy


class Face:
    """Represent a two-dimensional face and its absolute placement."""

    def __init__(self, type, dim):
        self.type: str = type
        self.dim: list = dim
        self.dim_abs: list = None

    def set_position_abs(self, x=0, y=0, z=0):
        if self.type == "BOX":
            self.dim_abs = [
                x + self.dim[0],
                y + self.dim[1],
                x + self.dim[2],
                y + self.dim[3],
            ]
        elif self.type == "CYLINDER":
            self.dim_abs = [
                x + self.dim[0],
                y + self.dim[1],
                self.dim[2],
            ]
        elif self.type == "POLYGON":
            self.dim_abs = [
                [[node[0] + x, node[1] + y] for node in polygon]
                for polygon in self.dim
            ]
        else:
            raise ValueError("unknown type")

    def copy(self):
        face_dup = Face(self.type, deepcopy(self.dim))
        face_dup.dim_abs = deepcopy(self.dim_abs)
        return face_dup

    def info(self, isAbs=False):
        if isAbs:
            return self.dict_abs()
        return self.dict()

    def dict(self):
        return {
            "type": self.type,
            "dim": self.dim,
            "dim_abs": self.dim_abs,
        }

    def dict_abs(self):
        return {
            "type": self.type,
            "dim": self.dim_abs,
        }
