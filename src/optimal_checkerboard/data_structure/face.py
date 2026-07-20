"""Face data structure for two-dimensional layout primitives.

A Face stores local dimensions for boxes, cylinders, and polygons, then
derives absolute coordinates when placed under an object origin.
"""

from copy import deepcopy
import math


class Face:
    """Represent a two-dimensional face and its absolute placement."""

    def __init__(self, type, dim):
        self.type: str = str(type).upper()
        self.dim: list = _normalize_face_dim(self.type, dim)
        self.dim_abs: list = None

    def set_position_abs(self, x=0, y=0, z=0):
        x = _finite_number(x, "face x offset")
        y = _finite_number(y, "face y offset")
        _finite_number(z, "face z offset")
        if self.type == "BOX":
            self.dim_abs = [
                _finite_number(x + self.dim[0], "absolute BOX coordinate"),
                _finite_number(y + self.dim[1], "absolute BOX coordinate"),
                _finite_number(x + self.dim[2], "absolute BOX coordinate"),
                _finite_number(y + self.dim[3], "absolute BOX coordinate"),
            ]
        elif self.type == "CYLINDER":
            self.dim_abs = [
                _finite_number(x + self.dim[0], "absolute CYLINDER coordinate"),
                _finite_number(y + self.dim[1], "absolute CYLINDER coordinate"),
                self.dim[2],
            ]
        elif self.type == "POLYGON":
            self.dim_abs = [
                [
                    [
                        _finite_number(
                            node[0] + x,
                            "absolute POLYGON coordinate",
                        ),
                        _finite_number(
                            node[1] + y,
                            "absolute POLYGON coordinate",
                        ),
                    ]
                    for node in polygon
                ]
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


def _finite_number(value, name):
    """Return one finite floating-point geometry coordinate."""
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return value


def _normalize_face_dim(face_type, dim):
    """Validate face coordinates and canonicalize rectangular bounds."""
    if face_type == "BOX":
        if len(dim) != 4:
            raise ValueError("BOX dim must be [xmin, ymin, xmax, ymax]")
        x1, y1, x2, y2 = (
            _finite_number(value, "BOX coordinate") for value in dim
        )
        xmin, xmax = sorted((x1, x2))
        ymin, ymax = sorted((y1, y2))
        if xmin >= xmax or ymin >= ymax:
            raise ValueError("BOX dim must have positive area")
        return [xmin, ymin, xmax, ymax]

    if face_type == "CYLINDER":
        if len(dim) != 3:
            raise ValueError("CYLINDER dim must be [cx, cy, radius]")
        cx, cy, radius = (
            _finite_number(value, "CYLINDER coordinate") for value in dim
        )
        if radius <= 0.0:
            raise ValueError("CYLINDER radius must be positive")
        return [cx, cy, radius]

    if face_type == "POLYGON":
        result = []
        for loop in dim:
            normalized_loop = []
            for point in loop:
                if len(point) < 2:
                    raise ValueError("POLYGON points must contain x and y")
                normalized_loop.append(
                    [
                        _finite_number(point[0], "POLYGON coordinate"),
                        _finite_number(point[1], "POLYGON coordinate"),
                    ]
                )
            result.append(normalized_loop)
        return result

    raise ValueError(f"unknown face type: {face_type}")
