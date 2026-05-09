"""Mesh data structure for local refinement controls.

A Mesh stores vertical bounds plus optional line or face constraints that can
be converted to absolute coordinates for meshing an object region.

"""

from copy import deepcopy

from .face import Face


class Mesh:
    """Represent mesh refinement information and absolute placement."""

    def __init__(self, begin, end, element_size=None, line=None, face=None):
        self.begin: float = begin
        self.end: float = end
        self.element_size: float = element_size
        self.line: list = line
        self.face: Face = face

        self.begin_abs = None
        self.end_abs = None
        self.line_abs = None

    def set_position_abs(self, x=0, y=0, z=0):
        self.begin_abs = self.begin + z
        self.end_abs = self.end + z

        if self.face is not None:
            self.face.set_position_abs(x, y, z)

        if self.line is not None:
            self.line_abs = [[x + node[0], y + node[1]] for node in self.line]

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
        else:
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
