"""Mesh data structure for local refinement controls.

A Mesh stores vertical bounds plus optional line or face constraints that can
be converted to absolute coordinates for meshing an object region.

"""

from copy import deepcopy
import math

from .face import Face


class Mesh:
    """Represent mesh refinement information and absolute placement."""

    def __init__(self, begin, end, element_size=None, line=None, face=None):
        self.begin: float = _finite_number(begin, "Mesh begin")
        self.end: float = _finite_number(end, "Mesh end")
        if element_size is None:
            self.element_size = None
        else:
            self.element_size = _finite_number(element_size, "Mesh element_size")
            if self.element_size <= 0.0:
                raise ValueError("Mesh element_size must be positive")
        self.line: list = _normalize_line(line)
        self.face: Face = face

        self.begin_abs = None
        self.end_abs = None
        self.line_abs = None

    def set_position_abs(self, x=0, y=0, z=0):
        x = _finite_number(x, "Mesh x offset")
        y = _finite_number(y, "Mesh y offset")
        z = _finite_number(z, "Mesh z offset")
        self.begin_abs = _finite_number(self.begin + z, "Mesh begin_abs")
        self.end_abs = _finite_number(self.end + z, "Mesh end_abs")

        if self.face is not None:
            self.face.set_position_abs(x, y, z)

        if self.line is not None:
            self.line_abs = [
                [
                    _finite_number(x + node[0], "absolute Mesh line coordinate"),
                    _finite_number(y + node[1], "absolute Mesh line coordinate"),
                ]
                for node in self.line
            ]

    def copy(self):
        face_dup = self.face.copy() if self.face is not None else None
        mesh_dup = Mesh(
            self.begin,
            self.end,
            self.element_size,
            deepcopy(self.line),
            face_dup,
        )
        mesh_dup.begin_abs = self.begin_abs
        mesh_dup.end_abs = self.end_abs
        mesh_dup.line_abs = deepcopy(self.line_abs)
        return mesh_dup

    def info(self, isAbs=False):
        if isAbs:
            return {
                "begin": self.begin_abs,
                "end": self.end_abs,
                "element_size": self.element_size,
                "line": self.line_abs,
                "face": self.face.info(isAbs=True) if self.face is not None else None,
            }
        return {
            "begin": self.begin,
            "end": self.end,
            "element_size": self.element_size,
            "line": self.line,
            "face": self.face.info() if self.face is not None else None,
            "begin_abs": self.begin_abs,
            "end_abs": self.end_abs,
            "line_abs": self.line_abs,
        }


def _finite_number(value, name):
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _normalize_line(line):
    if line is None:
        return None
    if len(line) != 2 or any(len(point) < 2 for point in line):
        raise ValueError("Mesh line must contain two xy endpoints")
    return [
        [
            _finite_number(point[0], "Mesh line coordinate"),
            _finite_number(point[1], "Mesh line coordinate"),
        ]
        for point in line
    ]
